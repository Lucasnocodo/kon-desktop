"""Stereo capture tests use synthetic pixels and never request camera access."""

import queue
import threading
import time

import cv2
import numpy as np
import pytest

from kon import stereo_capture as module
from kon.stereo_capture import CameraSource, CapturedFrame, DualCapture, FramePair, list_camera_sources


def pixels(value=0, width=16, height=12):
    return np.full((height, width, 3), value, dtype=np.uint8)


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = queue.Queue()
        self.ready = threading.Event()
        self.released = threading.Event()
        self.settings = []
        self.owner = None
        self.release_thread = None

    def isOpened(self):
        self.owner = threading.get_ident()
        return self.opened

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def read(self):
        assert self.owner == threading.get_ident()
        self.ready.set()
        item = self.frames.get(timeout=3)
        if isinstance(item, Exception):
            raise item
        return item

    def release(self):
        self.release_thread = threading.get_ident()
        self.released.set()

    def push(self, value=0, **kwargs):
        self.frames.put((True, pixels(value, **kwargs)))

    def finish(self):
        self.frames.put((False, None))


def setup_capture(**kwargs):
    cameras = [FakeCapture(), FakeCapture()]
    calls = []

    def factory(index, backend):
        calls.append((index, backend))
        return cameras[index]

    kwargs.setdefault("close_timeout_s", .05)
    capture = DualCapture(0, 1, capture_factory=factory, **kwargs)
    return capture, cameras, calls


def finish(capture, cameras):
    for camera in cameras:
        camera.finish()
    assert capture.close()


def wait_queued(capture, counts):
    with capture._condition:
        assert capture._condition.wait_for(
            lambda: all(len(q) >= count for q, count in zip(capture._queues, counts)) or capture.failure,
            timeout=1,
        )


def test_constructing_and_discovering_never_opens_camera(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No camera should be opened")

    monkeypatch.setattr(module.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(module.sys, "platform", "linux")
    capture = DualCapture(0, 1)
    assert not capture.running
    sources = list_camera_sources()
    assert [source.index for source in sources] == list(range(6))
    assert all(source.device_id == "" and "未驗證" in source.name for source in sources)
    assert capture.close()
    assert capture.closed


def test_native_discovery_returns_verified_mapping_without_probing(monkeypatch):
    expected = [CameraSource(0, "USB camera", "stable-a"), CameraSource(1, "iPhone", "stable-b")]
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.cv2, "__version__", "4.14.0")
    monkeypatch.setattr(module, "_avfoundation_sources", lambda: expected)
    assert list_camera_sources() == expected


def test_unknown_opencv_mapping_never_claims_device_identity(monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.cv2, "__version__", "5.0.0")
    monkeypatch.setattr(module, "_avfoundation_sources", lambda: pytest.fail("Unknown mapping"))
    assert all(not source.device_id for source in list_camera_sources())


def test_native_discovery_failure_has_explicit_unverified_fallback(monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.cv2, "__version__", "4.14.0")

    def unavailable():
        raise ImportError("PyObjC unavailable")

    monkeypatch.setattr(module, "_avfoundation_sources", unavailable)
    assert len(list_camera_sources()) == 6
    assert all(not source.device_id for source in list_camera_sources())


@pytest.mark.parametrize("a,b", [(0, 0), (-1, 1), (True, 1), (0.5, 1)])
def test_rejects_duplicate_or_invalid_indices(a, b):
    with pytest.raises(ValueError):
        DualCapture(a, b)


def test_pair_timestamp_conservatively_uses_older_input():
    pair = FramePair(CapturedFrame(10, pixels(), 1), CapturedFrame(10.015, pixels(), 8))
    assert pair.timestamp == 10
    assert pair.skew_s == pytest.approx(.015)


def test_nearest_pair_uses_each_image_once_and_drops_old_unmatched_frames():
    capture = DualCapture(0, 1)
    capture._queues[0].extend(CapturedFrame(t, pixels(), i) for i, t in enumerate([9.98, 10.01, 10.04], 1))
    capture._queues[1].extend(CapturedFrame(t, pixels(), i) for i, t in enumerate([10, 10.041], 1))
    pair = capture._pair_locked(10.05)
    assert (pair.primary.sequence, pair.secondary.sequence) == (3, 2)
    assert pair.skew_s == pytest.approx(.001)
    assert capture._pair_locked(10.05) is None
    assert not any(capture._queues)


def test_unmatched_timestamps_and_stale_samples_never_produce_a_pair():
    capture = DualCapture(0, 1)
    capture._queues[0].append(CapturedFrame(10, pixels(), 1))
    capture._queues[1].append(CapturedFrame(10.03, pixels(), 1))
    assert capture._pair_locked(10.04) is None
    capture._queues[0].append(CapturedFrame(9, pixels(), 2))
    assert capture._pair_locked(10.04) is None


def test_read_threads_request_modes_and_deliver_single_use_owned_pixels():
    capture, cameras, calls = setup_capture(max_skew_s=.1)
    try:
        capture.start()
        assert capture.running
        for camera in cameras:
            assert camera.ready.wait(1)
            assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in camera.settings
            assert (cv2.CAP_PROP_FRAME_HEIGHT, 720) in camera.settings
            assert (cv2.CAP_PROP_FPS, 30) in camera.settings
            assert (cv2.CAP_PROP_BUFFERSIZE, 1) in camera.settings
        original = pixels(17)
        cameras[0].frames.put((True, original))
        cameras[1].push(23)
        wait_queued(capture, (1, 1))
        original[:] = 99
        pair = capture.take()
        assert pair is not None
        assert pair.primary.sequence == pair.secondary.sequence == 1
        assert np.all(pair.primary.pixels == 17)
        assert np.all(pair.secondary.pixels == 23)
        assert capture.primary_size == capture.secondary_size == (16, 12)
        assert capture.take(timeout=0) is None
        assert len(calls) == 2
        capture.start()
        assert len(calls) == 2
    finally:
        finish(capture, cameras)
    assert all(camera.owner == camera.release_thread for camera in cameras)
    assert capture.closed
    assert capture.close()


def test_capture_queues_bound_memory_when_consumer_is_slow():
    capture, cameras, _ = setup_capture(max_age_s=3)
    try:
        capture.start()
        for camera in cameras:
            for i in range(20):
                camera.push(i)
        with capture._condition:
            assert capture._condition.wait_for(
                lambda: all(q and q[-1].sequence == 20 for q in capture._queues), timeout=1,
            )
            assert all(len(q) == 4 for q in capture._queues)
        pair = capture.take()
        assert pair is not None
        assert pair.primary.sequence >= 17
        assert pair.secondary.sequence >= 17
    finally:
        finish(capture, cameras)


def test_stalled_stream_fails_closed_instead_of_reusing_previous_pair():
    now = [100.]
    capture, cameras, _ = setup_capture(clock=lambda: now[0])
    try:
        capture.start()
        for camera in cameras:
            camera.push()
        wait_queued(capture, (1, 1))
        assert capture.take() is not None
        now[0] = 100.31
        assert capture.take(timeout=0) is None
        assert "沒有新影像" in capture.failure
        assert not capture.running
    finally:
        finish(capture, cameras)


def test_read_failure_stops_both_sources_and_discards_pending_images():
    capture, cameras, _ = setup_capture()
    try:
        capture.start()
        cameras[0].push()
        cameras[1].frames.put(RuntimeError("disconnected"))
        with capture._condition:
            assert capture._condition.wait_for(lambda: capture.failure is not None, timeout=1)
        assert "disconnected" in capture.failure
        assert capture.take(timeout=0) is None
    finally:
        finish(capture, cameras)


def test_source_size_change_requires_recalibration():
    capture, cameras, _ = setup_capture()
    try:
        capture.start()
        for camera in cameras:
            camera.push()
        wait_queued(capture, (1, 1))
        cameras[0].push(width=20)
        with capture._condition:
            assert capture._condition.wait_for(lambda: capture.failure is not None, timeout=1)
        assert "尺寸已變更" in capture.failure
        assert capture.take(timeout=0) is None
    finally:
        finish(capture, cameras)


def test_open_failure_releases_the_other_opened_camera_on_its_own_thread():
    first, second = FakeCapture(), FakeCapture(opened=False)
    second_entered = threading.Event()

    def factory(index, _backend):
        if index == 1:
            assert first.ready.wait(1)
            second_entered.set()
            return second
        return first

    capture = DualCapture(0, 1, capture_factory=factory, close_timeout_s=.02)
    try:
        with pytest.raises(RuntimeError, match="無法開啟鏡頭 1"):
            capture.start()
        assert second_entered.is_set()
        assert second.released.is_set()
        assert not first.released.is_set()  # No release races a blocked read.
    finally:
        finish(capture, (first, second))
    assert first.release_thread == first.owner
    assert second.release_thread == second.owner


def test_close_is_bounded_without_releasing_a_blocked_read():
    capture, cameras, _ = setup_capture(close_timeout_s=.03)
    capture.start()
    assert all(camera.ready.wait(1) for camera in cameras)
    started = time.monotonic()
    assert not capture.close()
    assert time.monotonic() - started < .2
    assert not any(camera.released.is_set() for camera in cameras)
    finish(capture, cameras)
    assert all(camera.owner == camera.release_thread for camera in cameras)


def test_start_timeout_is_bounded_even_if_camera_open_blocks():
    unblock = threading.Event()
    cameras = [FakeCapture(), FakeCapture()]

    def factory(index, _backend):
        assert unblock.wait(2)
        return cameras[index]

    capture = DualCapture(0, 1, capture_factory=factory, startup_timeout_s=.03, close_timeout_s=.02)
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="啟動逾時"):
            capture.start()
        assert time.monotonic() - started < .2
        assert not capture.closed
    finally:
        unblock.set()
        assert capture.close()
    assert all(camera.released.is_set() for camera in cameras)


def test_verified_identity_mismatch_rejects_open_without_touching_hardware(monkeypatch):
    monkeypatch.setattr(module, "list_camera_sources", lambda: [CameraSource(0, "Other", "new"), CameraSource(1, "Phone", "phone")])
    called = []

    def factory(*args):
        called.append(args)
        return FakeCapture(opened=False)

    capture = DualCapture(CameraSource(0, "USB", "old"), CameraSource(1, "Phone", "phone"), capture_factory=factory)
    with pytest.raises(RuntimeError, match="裝置已變更"):
        capture.start()
    assert capture.closed
    assert not any(index == 0 for index, _backend in called)


def test_native_discovery_matches_opencv_video_muxed_and_id_sort(monkeypatch):
    from types import SimpleNamespace
    import sys

    class NativeString(str):
        @classmethod
        def stringWithString_(cls, value):
            return cls(value)

        def compare_(self, other):
            return (self > other) - (self < other)

    class Device:
        def __init__(self, identifier, name):
            self.identifier, self.name = identifier, name

        def uniqueID(self):
            return self.identifier

        def localizedName(self):
            return self.name

    requested_media = []

    class DeviceClass:
        @staticmethod
        def devicesWithMediaType_(media):
            requested_media.append(media)
            return ([Device("b", "iPhone"), Device("a", "USB")]
                    if media == "vide" else [Device("c", "Muxed camera")])

    monkeypatch.setitem(sys.modules, "objc", SimpleNamespace(
        loadBundle=lambda *_args, **_kwargs: None,
        lookUpClass=lambda name: DeviceClass if name == "AVCaptureDevice" else None,
    ))
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSString=NativeString))
    assert module._avfoundation_sources() == [
        CameraSource(0, "USB", "a"), CameraSource(1, "iPhone", "b"),
        CameraSource(2, "Muxed camera", "c"),
    ]
    assert requested_media == ["vide", "muxx"]


def test_no_initial_frames_fails_after_startup_grace():
    now = [100.]
    capture, cameras, _ = setup_capture(clock=lambda: now[0])
    try:
        capture.start()
        assert capture.take(timeout=0) is None
        assert capture.failure is None
        now[0] = 101.01
        assert capture.take(timeout=0) is None
        assert "沒有新影像" in capture.failure
    finally:
        finish(capture, cameras)


def test_repeated_failed_reads_stop_both_sources():
    capture, cameras, _ = setup_capture()
    try:
        capture.start()
        for _ in range(5):
            cameras[0].finish()
        with capture._condition:
            assert capture._condition.wait_for(lambda: capture.failure is not None, timeout=1)
        assert "無法讀取影像" in capture.failure
        assert capture.take(timeout=0) is None
    finally:
        finish(capture, cameras)


def test_identity_change_while_opening_releases_camera_before_any_read(monkeypatch):
    calls = [0]

    def discover():
        calls[0] += 1
        return [CameraSource(0, "USB", "expected" if calls[0] == 1 else "changed")]

    monkeypatch.setattr(module, "list_camera_sources", discover)
    cameras = [FakeCapture(), FakeCapture()]
    capture = DualCapture(CameraSource(0, "USB", "expected"), 1,
                          capture_factory=lambda index, _backend: cameras[index],
                          close_timeout_s=.02)
    try:
        with pytest.raises(RuntimeError, match="裝置已變更"):
            capture.start()
        assert cameras[0].released.is_set()
        assert not cameras[0].ready.is_set()
    finally:
        finish(capture, cameras)
