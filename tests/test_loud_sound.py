import queue
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kon.speech import (
    BLOCK_FRAMES,
    LOUD_SOUND_MARKER,
    LoudSoundDetector,
    SpeechWorker,
)


def audio(amplitude=0.1):
    return np.full(BLOCK_FRAMES, amplitude, dtype=np.float32)


class LoudSoundDetectorTests(unittest.TestCase):
    def test_three_blocks_trigger_immediately_and_continuous_sound_stays_latched(self):
        detector = LoudSoundDetector()
        self.assertIsNone(detector.feed(audio(), 100.02))
        self.assertIsNone(detector.feed(audio(), 100.04))
        self.assertEqual(detector.feed(audio(), 100.06), 100.06)
        for index in range(500):
            self.assertIsNone(detector.feed(audio(), 100.08 + index * 0.02))

    def test_short_quiet_gap_keeps_latch_but_full_quiet_rearms(self):
        detector = LoudSoundDetector()
        for stamp in (100.02, 100.04, 100.06):
            result = detector.feed(audio(), stamp)
        self.assertEqual(result, 100.06)
        for index in range(7):
            self.assertIsNone(detector.feed(audio(0), 100.08 + index * 0.02))
        self.assertIsNone(detector.feed(audio(), 100.22))
        for index in range(8):
            self.assertIsNone(detector.feed(audio(0), 100.24 + index * 0.02))
        self.assertIsNone(detector.feed(audio(), 100.4))
        self.assertIsNone(detector.feed(audio(), 100.42))
        self.assertEqual(detector.feed(audio(), 100.44), 100.44)

    def test_low_volume_isolated_impulse_and_disconnected_blocks_do_not_trigger(self):
        detector = LoudSoundDetector()
        for index in range(200):
            self.assertIsNone(detector.feed(audio(0.02), 100 + index * 0.02))
        self.assertIsNone(detector.feed(audio(0.8), 104.0))
        for index in range(8):
            self.assertIsNone(detector.feed(audio(0), 104.02 + index * 0.02))
        for index in range(20):
            self.assertIsNone(detector.feed(audio(), 105.0 + index * 0.06))
            self.assertIsNone(detector.feed(audio(), 105.02 + index * 0.06))
            self.assertIsNone(detector.feed(audio(0), 105.04 + index * 0.06))

    def test_reset_and_invalid_audio_cannot_complete_an_old_partial_burst(self):
        for invalid in (audio(float("nan")), np.empty(0, dtype=np.float32)):
            with self.subTest(invalid=invalid):
                detector = LoudSoundDetector()
                detector.feed(audio(), 100.02)
                detector.feed(audio(), 100.04)
                self.assertIsNone(detector.feed(invalid, 100.06))
                self.assertIsNone(detector.feed(audio(), 100.08))
                detector.reset()
                self.assertIsNone(detector.feed(audio(), 100.10))
                self.assertIsNone(detector.feed(audio(), 100.12))
                self.assertEqual(detector.feed(audio(), 100.14), 100.14)


class LoudSoundWorkerTests(unittest.TestCase):
    def make_worker(self, project_root=None):
        worker = SpeechWorker(project_root, trigger_mode="loud_sound")
        with patch("kon.speech.time.monotonic", return_value=100.0):
            worker.set_active(True)
        return worker

    def block(self, worker, stamp, amplitude=0.1, *, arrival=None, generation=None):
        return lambda: (
            worker._generation if generation is None else generation,
            audio(amplitude), stamp, stamp + 0.05 if arrival is None else arrival,
        )

    def run_script(self, worker, script):
        """Run the real worker loop with deterministic capture and arrival clocks."""
        entries = iter(script)
        clock = [100.0]

        def receive(block=True, timeout=None):
            if not block:
                raise queue.Empty
            try:
                entry = next(entries)
            except StopIteration:
                worker.stop()
                raise queue.Empty
            while callable(entry):
                entry = entry()
            generation, samples, stamp, arrival = entry
            clock[0] = arrival
            return generation, samples, stamp

        with patch("kon.speech.time.monotonic", side_effect=lambda: clock[0]), \
                patch("kon.speech.resolve_microphone", return_value=SimpleNamespace(index=0, name="Test mic")), \
                patch("kon.speech.sd.InputStream") as microphone, \
                patch.object(worker._audio, "get", side_effect=receive):
            worker.run()
        microphone.assert_called_once()

    def test_loud_capture_emits_original_timestamp_without_whisper_or_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = self.make_worker(temporary)
            keywords, transcripts, errors = [], [], []
            worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
            worker.transcript.connect(transcripts.append)
            worker.error.connect(errors.append)
            self.assertFalse(worker.cli_path.is_file())
            self.assertFalse(worker.model_path.is_file())
            script = [self.block(worker, 100.02 + index * 0.02, amplitude=0) for index in range(8)]
            script += [self.block(worker, 100.18 + index * 0.02) for index in range(90)]
            with patch.object(worker, "_transcribe") as transcribe:
                self.run_script(worker, script)
            self.assertEqual(keywords, [(100.18 + 2 * 0.02, LOUD_SOUND_MARKER)])
            self.assertEqual(transcripts, [])
            self.assertEqual(errors, [])
            transcribe.assert_not_called()

    def test_stale_overflow_and_old_generation_reset_partial_bursts(self):
        for interrupt in ("stale", "overflow", "rearm"):
            with self.subTest(interrupt=interrupt):
                worker = self.make_worker()
                keywords = []
                worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
                old_generation = worker._generation

                def interrupted_block():
                    if interrupt == "stale":
                        return self.block(worker, 100.22, arrival=101.0)
                    if interrupt == "overflow":
                        worker._overflow.set()
                    else:
                        worker.set_active(False)
                        worker.set_active(True)
                    return self.block(worker, 101.02, generation=old_generation)

                self.run_script(worker, [
                    *(self.block(worker, 100.02 + index * 0.02, amplitude=0) for index in range(8)),
                    self.block(worker, 100.18), self.block(worker, 100.20),
                    interrupted_block,
                    *(self.block(worker, 101.04 + index * 0.02, amplitude=0) for index in range(8)),
                    self.block(worker, 101.20), self.block(worker, 101.22),
                    self.block(worker, 101.24),
                ])
                self.assertEqual(keywords, [(101.24, LOUD_SOUND_MARKER)])

    def test_disarm_between_detection_and_publication_discards_event(self):
        worker = self.make_worker()
        keywords = []
        worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
        real_feed = LoudSoundDetector.feed

        def cancel_at_detection(detector, samples, stamp):
            result = real_feed(detector, samples, stamp)
            if result is not None:
                worker.set_active(False)
            return result

        with patch.object(LoudSoundDetector, "feed", cancel_at_detection):
            self.run_script(worker, [
                *(self.block(worker, 100.02 + index * 0.02, amplitude=0) for index in range(8)),
                *(self.block(worker, 100.18 + index * 0.02) for index in range(3)),
            ])
        self.assertEqual(keywords, [])

    def test_fresh_gesture_can_trigger_again_within_old_phrase_cooldown(self):
        worker = self.make_worker()
        keywords = []
        worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))

        def new_gesture():
            worker.set_active(False)
            worker.set_active(True)
            return self.block(worker, 100.92, amplitude=0)

        self.run_script(worker, [
            *(self.block(worker, 100.02 + index * 0.02, amplitude=0) for index in range(8)),
            self.block(worker, 100.18), self.block(worker, 100.20), self.block(worker, 100.22),
            new_gesture,
            *(self.block(worker, 100.94 + index * 0.02, amplitude=0) for index in range(7)),
            self.block(worker, 101.08), self.block(worker, 101.10), self.block(worker, 101.12),
        ])
        self.assertEqual(keywords, [(100.22, LOUD_SOUND_MARKER), (101.12, LOUD_SOUND_MARKER)])

    def test_same_arm_rearms_after_quiet_without_phrase_cooldown(self):
        worker = self.make_worker()
        keywords = []
        worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
        self.run_script(worker, [
            *(self.block(worker, 100.02 + index * 0.02, amplitude=0) for index in range(8)),
            self.block(worker, 100.18), self.block(worker, 100.20), self.block(worker, 100.22),
            *(self.block(worker, 100.24 + index * 0.02, amplitude=0) for index in range(8)),
            self.block(worker, 100.40), self.block(worker, 100.42), self.block(worker, 100.44),
        ])
        self.assertEqual(keywords, [(100.22, LOUD_SOUND_MARKER), (100.44, LOUD_SOUND_MARKER)])

    def test_mode_validation_and_reconfiguration_preserve_selected_mode(self):
        worker = SpeechWorker(trigger_mode="loud_sound")
        worker.configure_trigger("吃掉")
        self.assertEqual(worker.trigger_mode, "loud_sound")
        worker.configure_trigger("kon", trigger_mode="phrase")
        self.assertEqual(worker.trigger_mode, "phrase")
        for mode in (None, True, "other"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                SpeechWorker(trigger_mode=mode)
        with self.assertRaises(ValueError):
            worker.configure_trigger("kon", trigger_mode="other")
        self.assertEqual(worker.trigger_mode, "phrase")
        self.assertEqual(SpeechWorker().trigger_mode, "phrase")

    def test_asr_result_never_triggers_ordinary_loud_mode(self):
        worker = self.make_worker()
        keywords, transcripts = [], []
        worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
        worker.transcript.connect(transcripts.append)
        worker._publish_result("kon", 100.1, worker._generation)
        self.assertEqual(keywords, [])
        self.assertEqual(transcripts, [])

    def test_pronunciation_capture_uses_exact_asr_text_and_never_loud_trigger(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = self.make_worker(temporary)
            worker.cli_path = Path(temporary) / "whisper"
            worker.model_path = Path(temporary) / "model.bin"
            worker.cli_path.touch()
            worker.model_path.touch()
            samples, keywords, transcripts, errors = [], [], [], []
            worker.sample.connect(lambda request, text: samples.append((request, text)))
            worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
            worker.transcript.connect(transcripts.append)
            worker.error.connect(errors.append)
            with patch("kon.speech.time.monotonic", return_value=100.0):
                worker.request_sample(7, language="zh")
            script = [self.block(worker, 100.02 + index * 0.02) for index in range(10)]
            script += [self.block(worker, 100.22 + index * 0.02, amplitude=0) for index in range(20)]
            with patch.object(worker, "_transcribe", return_value="（CON）") as transcribe:
                self.run_script(worker, script)
            self.assertEqual(samples, [(7, "（CON）")])
            self.assertEqual(keywords, [])
            self.assertEqual(transcripts, [])
            self.assertEqual(errors, [])
            transcribe.assert_called_once()

    def test_missing_sample_resources_fail_sample_and_keep_loud_worker_usable(self):
        for missing in ("cli", "model"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                worker = self.make_worker(temporary)
                if missing == "model":
                    worker.cli_path = Path(temporary) / "whisper"
                    worker.cli_path.touch()
                failures, keywords, errors = [], [], []
                worker.sample_failed.connect(lambda request, message: failures.append((request, message)))
                worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
                worker.error.connect(errors.append)
                with patch("kon.speech.time.monotonic", return_value=100.0):
                    worker.request_sample(8)
                script = [self.block(worker, 100.02 + index * 0.02) for index in range(10)]
                script += [self.block(worker, 100.22 + index * 0.02, amplitude=0) for index in range(17)]

                def resume_loud():
                    self.assertFalse(worker._active)
                    worker.set_active(True)
                    return self.block(worker, 100.62, amplitude=0)

                script += [
                    resume_loud,
                    *(self.block(worker, 100.64 + index * 0.02, amplitude=0) for index in range(7)),
                    self.block(worker, 100.78), self.block(worker, 100.80), self.block(worker, 100.82),
                ]
                with patch.object(worker, "_transcribe") as transcribe:
                    self.run_script(worker, script)
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0][0], 8)
                self.assertIn("Whisper" if missing == "cli" else "模型", failures[0][1])
                self.assertEqual(keywords, [(100.82, LOUD_SOUND_MARKER)])
                self.assertEqual(errors, [])
                transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
