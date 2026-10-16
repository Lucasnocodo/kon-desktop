"""Bounded, independently read camera streams for stereo reconstruction.

Timestamps are monotonic HOST_RECEIVE times immediately after ``read()``.
They are not sensor exposure times: a small skew cannot establish hardware
synchronization or compensate for Continuity Camera's transport latency.
Constructors and device discovery never open a capture session. Only start()
opens cameras, and each daemon producer exclusively owns open/read/release.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cmp_to_key
import math
import re
import sys
import threading
import time
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CameraSource:
    index: int
    name: str
    device_id: str = ""


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    timestamp: float
    pixels: np.ndarray
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class FramePair:
    primary: CapturedFrame
    secondary: CapturedFrame

    @property
    def timestamp(self) -> float:
        """Use the older input for conservative downstream freshness checks."""
        return min(self.primary.timestamp, self.secondary.timestamp)

    @property
    def skew_s(self) -> float:
        return abs(self.primary.timestamp - self.secondary.timestamp)


def _avfoundation_sources() -> list[CameraSource]:
    """Mirror OpenCV's verified AVFoundation index ordering, without inputs.

    OpenCV 4.10–4.14 cap_avfoundation_mac.mm concatenates the legacy video
    and muxed lists, then sorts by NSString uniqueID comparison. Using a
    newer DiscoverySession here would enumerate a different index order.
    Source: https://github.com/opencv/opencv/blob/4.14.0/modules/videoio/src/cap_avfoundation_mac.mm
    """
    import objc
    from Foundation import NSString

    objc.loadBundle(
        "AVFoundation", {},
        bundle_path="/System/Library/Frameworks/AVFoundation.framework",
    )
    device_class = objc.lookUpClass("AVCaptureDevice")
    devices = list(device_class.devicesWithMediaType_("vide") or ())
    devices += list(device_class.devicesWithMediaType_("muxx") or ())

    def compare(first, second):
        a = NSString.stringWithString_(str(first.uniqueID()))
        b = NSString.stringWithString_(str(second.uniqueID()))
        return int(a.compare_(b))

    devices.sort(key=cmp_to_key(compare))
    identifiers = [str(device.uniqueID()) for device in devices]
    # Ambiguous IDs cannot identify two separate calibrated sources reliably.
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("鏡頭識別資訊重複或不完整")
    return [CameraSource(index, str(device.localizedName()), identifiers[index])
            for index, device in enumerate(devices)]


def list_camera_sources() -> list[CameraSource]:
    """Read device metadata only; never probe indices or request camera access.

    Empty device_id explicitly means an unverified manual index. Such choices
    remain useful when native enumeration is unavailable, but must not be
    presented as a stable device identity for saved calibration reuse.
    """
    version = re.match(r"^(\d+)\.(\d+)\.", cv2.__version__)
    known_mapping = version is not None and int(version[1]) == 4 and 10 <= int(version[2]) <= 14
    if sys.platform == "darwin" and known_mapping:
        try:
            return _avfoundation_sources()
        except Exception:
            pass
    return [CameraSource(index, f"鏡頭 {index}（名稱與裝置識別未驗證）") for index in range(6)]


class DualCapture:
    """Two small queues with nearest-time, single-use frame pairing.

    start() waits startup_timeout_s for opening, then uses the bounded close
    wait as well if startup fails.
    close() waits at most close_timeout_s for both producers together; False
    means a driver call is still completing. It never releases a camera while
    its owner is reading it. Repeated close() may finish that pending cleanup.
    There is intentionally no silent fallback to a single camera.
    """

    def __init__(
        self, primary_source: int | CameraSource, secondary_source: int | CameraSource,
        *, max_skew_s: float = .020, max_age_s: float = .3,
        width: int = 1280, height: int = 720,
        stop_event: threading.Event | None = None,
        capture_factory: Callable | None = None,
        clock: Callable[[], float] | None = None,
        startup_timeout_s: float = 5.0, close_timeout_s: float = 2.0,
    ) -> None:
        self.primary_source = self._source(primary_source)
        self.secondary_source = self._source(secondary_source)
        if self.primary_source.index == self.secondary_source.index:
            raise ValueError("雙鏡頭必須選擇不同的鏡頭編號")
        if (self.primary_source.device_id and
                self.primary_source.device_id == self.secondary_source.device_id):
            raise ValueError("雙鏡頭必須選擇不同的裝置")
        if (not math.isfinite(max_skew_s) or max_skew_s <= 0 or
                not math.isfinite(max_age_s) or max_age_s <= 0 or max_skew_s > max_age_s):
            raise ValueError("雙鏡頭時間容許值無效")
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
            raise ValueError("鏡頭尺寸無效")
        if (not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0 or
                not math.isfinite(close_timeout_s) or close_timeout_s < 0):
            raise ValueError("鏡頭啟停等待時間無效")
        self.max_skew_s = float(max_skew_s)
        self.max_age_s = float(max_age_s)
        self.width, self.height = width, height
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._factory = capture_factory if capture_factory is not None else cv2.VideoCapture
        self._clock = clock if clock is not None else time.monotonic
        self._startup_timeout = startup_timeout_s
        self._close_timeout = close_timeout_s
        self._condition = threading.Condition()
        self._queues: tuple[deque[CapturedFrame], deque[CapturedFrame]] = (deque(maxlen=4), deque(maxlen=4))
        self._threads: list[threading.Thread] = []
        self._ready = [False, False]
        self._ready_at: list[float | None] = [None, None]
        self._last_received: list[float | None] = [None, None]
        self._sizes: list[tuple[int, int] | None] = [None, None]
        self._failure: str | None = None
        self._status = "雙鏡頭尚未啟動"
        self._started = False
        self._close_requested = False

    @staticmethod
    def _source(value: int | CameraSource) -> CameraSource:
        source = value if isinstance(value, CameraSource) else CameraSource(value, f"鏡頭 {value}")
        if isinstance(source.index, bool) or not isinstance(source.index, int) or source.index < 0:
            raise ValueError("請指定非負的鏡頭編號")
        return source

    @property
    def failure(self) -> str | None:
        with self._condition:
            return self._failure

    @property
    def status(self) -> str:
        with self._condition:
            return self._status

    @property
    def primary_size(self) -> tuple[int, int] | None:
        with self._condition:
            return self._sizes[0]

    @property
    def secondary_size(self) -> tuple[int, int] | None:
        with self._condition:
            return self._sizes[1]

    @property
    def running(self) -> bool:
        return self._started and not self.stop_event.is_set() and any(t.is_alive() for t in self._threads)

    @property
    def closed(self) -> bool:
        return self._close_requested and not any(t.is_alive() for t in self._threads)

    def _fail(self, message: str) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = message
                self._status = message
            self.stop_event.set()
            for queue in self._queues:
                queue.clear()
            self._condition.notify_all()

    @staticmethod
    def _verify_source(source: CameraSource) -> None:
        if not source.device_id:
            return
        current = {device.index: device.device_id for device in list_camera_sources()}
        if current.get(source.index) != source.device_id:
            raise RuntimeError(f"鏡頭 {source.index} 的裝置已變更；請重新選擇並校正雙鏡頭")

    def start(self) -> None:
        if self._close_requested or self.stop_event.is_set():
            raise RuntimeError("雙鏡頭已停止，請建立新的拍攝工作")
        if self._started:
            if self.failure:
                raise RuntimeError(self.failure)
            return
        self._started = True
        self._status = "正在開啟雙鏡頭…"
        for position, source in enumerate((self.primary_source, self.secondary_source)):
            thread = threading.Thread(target=self._produce, args=(position, source),
                                      name=f"KON stereo camera {source.index}", daemon=True)
            self._threads.append(thread)
            thread.start()
        deadline = time.monotonic() + self._startup_timeout
        with self._condition:
            while not all(self._ready) and not self._failure and not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            ready = all(self._ready)
        if not ready or self.stop_event.is_set():
            if self.failure is None:
                self._fail("雙鏡頭啟動逾時或已取消；請檢查鏡頭連接與權限")
            self.close()
            raise RuntimeError(self.failure)
        with self._condition:
            self._status = "雙鏡頭已啟動；以電腦收到影像的時間配對，並非同步曝光"

    def _produce(self, position: int, source: CameraSource) -> None:
        capture = None
        try:
            self._verify_source(source)
            if self.stop_event.is_set():
                return
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            capture = self._factory(source.index, backend)
            if not capture.isOpened():
                raise RuntimeError(f"無法開啟鏡頭 {source.index}；請檢查權限與是否被其他程式占用")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._verify_source(source)
            with self._condition:
                self._ready[position] = True
                self._ready_at[position] = self._clock()
                self._condition.notify_all()
            sequence = failures = 0
            while not self.stop_event.is_set():
                ok, pixels = capture.read()
                timestamp = self._clock()
                if self.stop_event.is_set():
                    break
                if not ok or pixels is None:
                    failures += 1
                    if failures >= 5:
                        raise RuntimeError(f"鏡頭 {source.index} 無法讀取影像；雙鏡頭瞄準已暫停")
                    self.stop_event.wait(.03)
                    continue
                if (not isinstance(pixels, np.ndarray) or pixels.ndim != 3 or
                        pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0 or
                        not math.isfinite(timestamp)):
                    raise RuntimeError(f"鏡頭 {source.index} 傳回無效影像")
                size = (pixels.shape[1], pixels.shape[0])
                failures = 0
                sequence += 1
                # Protect consumers even if a backend reuses its read buffer.
                frame = CapturedFrame(timestamp, pixels.copy(), sequence)
                with self._condition:
                    if self._sizes[position] is not None and self._sizes[position] != size:
                        raise RuntimeError(f"鏡頭 {source.index} 的影像尺寸已變更；請重新校正雙鏡頭")
                    self._sizes[position] = size
                    self._last_received[position] = timestamp
                    self._queues[position].append(frame)
                    self._condition.notify_all()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._fail(f"雙鏡頭讀取失敗：{exc}")
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    self._fail(f"鏡頭 {source.index} 關閉失敗：{exc}")
            with self._condition:
                self._condition.notify_all()

    def _pair_locked(self, now: float) -> FramePair | None:
        for queue in self._queues:
            while queue and now - queue[0].timestamp > self.max_age_s:
                queue.popleft()
        first, second = self._queues
        if not first or not second:
            return None
        # At most 4×4 candidates. Prefer newer samples when skews tie.
        candidates = [(abs(a.timestamp - b.timestamp), -min(a.timestamp, b.timestamp), i, j)
                      for i, a in enumerate(first) for j, b in enumerate(second)]
        skew, _negative_timestamp, i, j = min(candidates)
        if skew <= self.max_skew_s + 1e-9:
            primary, secondary = first[i], second[j]
            for _ in range(i + 1):
                first.popleft()
            for _ in range(j + 1):
                second.popleft()
            return FramePair(primary, secondary)
        # A frame older than the opposite stream's earliest frame cannot be
        # matched by a later arrival. Drop it without ever reusing samples.
        while first and second and first[0].timestamp < second[0].timestamp - self.max_skew_s:
            first.popleft()
        while first and second and second[0].timestamp < first[0].timestamp - self.max_skew_s:
            second.popleft()
        return None

    def take(self, timeout: float = .05) -> FramePair | None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("影像等待時間無效")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._started and not self.stop_event.is_set() and not self._failure:
                now = self._clock()
                for position in (0, 1):
                    received = self._last_received[position]
                    ready_at = self._ready_at[position]
                    expired = (received is not None and now - received > self.max_age_s)
                    never_arrived = received is None and ready_at is not None and now - ready_at > max(1.0, self.max_age_s)
                    if expired or never_arrived:
                        source = (self.primary_source, self.secondary_source)[position]
                        self._fail(f"鏡頭 {source.index} 沒有新影像；雙鏡頭瞄準已暫停")
                        return None
                pair = self._pair_locked(now)
                if pair is not None:
                    return pair
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, .05))
        return None

    def close(self) -> bool:
        self._close_requested = True
        self.stop_event.set()
        with self._condition:
            for queue in self._queues:
                queue.clear()
            self._condition.notify_all()
        deadline = time.monotonic() + self._close_timeout
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))
        closed = not any(thread.is_alive() for thread in self._threads)
        with self._condition:
            if self._failure is None:
                self._status = "雙鏡頭已關閉" if closed else "鏡頭驅動仍在結束讀取，請稍候"
        return closed
