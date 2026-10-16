import queue
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kon.speech import BLOCK_FRAMES, LOUD_SOUND_MARKER, SpeechWorker


class ContinuousAudioTests(unittest.TestCase):
    def make_worker(self, mode="loud_sound"):
        worker = SpeechWorker(trigger_mode=mode)
        with patch("kon.speech.time.monotonic", return_value=100.0):
            worker.set_active(True)
        return worker

    def run_audio(self, worker, actions):
        """Feed real detector/VAD logic without opening a physical microphone."""
        clock = [100.0]
        entries = iter(actions)
        keywords, transcripts, errors, transcriptions = [], [], [], []
        worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
        worker.transcript.connect(transcripts.append)
        worker.error.connect(errors.append)

        def receive(block=True, timeout=None):
            if not block:
                raise queue.Empty
            while True:
                try:
                    entry = next(entries)
                except StopIteration:
                    worker.stop()
                    raise queue.Empty
                if callable(entry):
                    entry(worker)
                    continue
                amplitude, count, options = entry
                if count != 1:
                    raise AssertionError("Expand block counts before running")
                clock[0] += options.get("advance", 0.02)
                stamp = clock[0] - options.get("age", 0.0)
                generation = options.get("generation", worker._generation)
                samples = np.full(BLOCK_FRAMES, amplitude, dtype=np.float32)
                return generation, samples, stamp

        def transcribe(chunk, generation):
            transcriptions.append(chunk)
            return "kon"

        with patch("kon.speech.time.monotonic", side_effect=lambda: clock[0]), \
                patch("kon.speech.resolve_microphone", return_value=SimpleNamespace(index=0, name="Test mic")), \
                patch("kon.speech.sd.InputStream"), \
                patch.object(worker._audio, "get", side_effect=receive), \
                patch.object(worker, "_require_recognizer"), \
                patch.object(worker, "_transcribe", side_effect=transcribe):
            worker.run()
        self.assertEqual(errors, [])
        return keywords, transcripts, transcriptions

    @staticmethod
    def blocks(amplitude, count, **options):
        return [(amplitude, 1, options) for _ in range(count)]

    @staticmethod
    def rearm(worker):
        worker.set_active(False)
        worker.set_active(True)

    def test_long_audio_present_at_each_arm_is_ignored_in_both_modes(self):
        for mode in ("loud_sound", "phrase"):
            with self.subTest(mode=mode):
                worker = self.make_worker(mode)
                keywords, transcripts, chunks = self.run_audio(worker, [
                    *self.blocks(0.1, 200), self.rearm,
                    *self.blocks(0.1, 200),
                ])
                self.assertEqual((keywords, transcripts, chunks), ([], [], []))

    def test_each_new_target_needs_quiet_and_a_new_burst(self):
        worker = self.make_worker()
        keywords, _, _ = self.run_audio(worker, [
            *self.blocks(0, 8), *self.blocks(0.1, 3),
            self.rearm, *self.blocks(0.1, 40),
            *self.blocks(0, 7), *self.blocks(0.1, 3),
            *self.blocks(0, 8), *self.blocks(0.1, 3),
        ])
        self.assertEqual([text for _, text in keywords], [LOUD_SOUND_MARKER] * 2)
        self.assertAlmostEqual(keywords[0][0], 100.22)
        self.assertAlmostEqual(keywords[1][0], 101.44)

    def test_idempotent_active_does_not_restart_quiet_progress(self):
        worker = self.make_worker()
        keywords, _, _ = self.run_audio(worker, [
            *self.blocks(0, 4), lambda worker: worker.set_active(True),
            *self.blocks(0, 4), *self.blocks(0.1, 3),
        ])
        self.assertEqual(len(keywords), 1)

    def test_quiet_while_inactive_does_not_arm_the_next_target(self):
        worker = self.make_worker()

        def inactive_quiet(worker):
            worker.set_active(False)
            for _ in range(8):
                worker._microphone_callback(
                    np.zeros((BLOCK_FRAMES, 1), dtype=np.float32), BLOCK_FRAMES,
                    SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.02), None,
                )
            self.assertTrue(worker._audio.empty())
            worker.set_active(True)

        keywords, _, _ = self.run_audio(worker, [
            *self.blocks(0, 8), *self.blocks(0.1, 3), inactive_quiet,
            *self.blocks(0.1, 20), *self.blocks(0, 8), *self.blocks(0.1, 3),
        ])
        self.assertEqual(len(keywords), 2)
        self.assertAlmostEqual(keywords[1][0], 100.84)

    def test_noise_above_loud_trigger_threshold_resets_the_quiet_interval(self):
        worker = self.make_worker()
        keywords, _, _ = self.run_audio(worker, [
            *self.blocks(0, 7), *self.blocks(0.03, 1),
            *self.blocks(0, 7), *self.blocks(0.1, 3),
            *self.blocks(0, 8), *self.blocks(0.1, 3),
        ])
        self.assertEqual(len(keywords), 1)
        self.assertAlmostEqual(keywords[0][0], 100.58)

    def test_low_background_noise_is_quiet_for_loud_mode_but_not_phrase_mode(self):
        for mode in ("loud_sound", "phrase"):
            with self.subTest(mode=mode):
                worker = self.make_worker(mode)
                keywords, transcripts, chunks = self.run_audio(worker, [
                    *self.blocks(0.02, 8), *self.blocks(0.1, 10),
                    *self.blocks(0, 20),
                ])
                self.assertEqual(len(keywords), 1 if mode == "loud_sound" else 0)
                self.assertEqual((transcripts, chunks), ([], []))

    def test_mode_change_updates_quiet_threshold_and_resets_previous_progress(self):
        worker = self.make_worker()

        def use_phrase(worker):
            worker.set_active(False)
            worker.configure_trigger("kon", trigger_mode="phrase")
            worker.set_active(True)

        def use_loud(worker):
            worker.set_active(False)
            worker.configure_trigger("kon", trigger_mode="loud_sound")
            worker.set_active(True)

        keywords, transcripts, chunks = self.run_audio(worker, [
            *self.blocks(0.02, 8), *self.blocks(0.1, 3),
            use_phrase,
            *self.blocks(0.02, 8), *self.blocks(0.1, 10), *self.blocks(0, 20),
            *self.blocks(0.1, 10), *self.blocks(0, 20),
            use_loud, *self.blocks(0.02, 8), *self.blocks(0.1, 3),
        ])
        self.assertEqual([text for _, text in keywords], [LOUD_SOUND_MARKER, "kon", LOUD_SOUND_MARKER])
        self.assertEqual(transcripts, ["kon"])
        self.assertEqual(len(chunks), 1)

    def test_missing_or_invalid_audio_cannot_supply_the_quiet_interval(self):
        for interrupt in ("gap", "nan", "stale", "overflow", "old_generation"):
            with self.subTest(interrupt=interrupt):
                worker = self.make_worker()
                if interrupt == "gap":
                    interruption = self.blocks(0, 1, advance=0.5)
                elif interrupt == "nan":
                    interruption = self.blocks(float("nan"), 1)
                elif interrupt == "stale":
                    interruption = self.blocks(0, 1, age=1.0)
                elif interrupt == "overflow":
                    interruption = [lambda worker: worker._overflow.set(), *self.blocks(0, 1)]
                else:
                    interruption = self.blocks(0, 1, generation=worker._generation - 1)
                keywords, _, _ = self.run_audio(worker, [
                    *self.blocks(0, 7), *interruption,
                    *self.blocks(0, 6), *self.blocks(0.1, 3),
                    *self.blocks(0, 8), *self.blocks(0.1, 3),
                ])
                self.assertEqual(len(keywords), 1)

    def test_overflow_after_a_burst_cannot_turn_its_continuation_into_a_new_one(self):
        worker = self.make_worker()
        keywords, _, _ = self.run_audio(worker, [
            *self.blocks(0, 8), *self.blocks(0.1, 3),
            lambda worker: worker._overflow.set(), *self.blocks(0.1, 50),
            *self.blocks(0, 8), *self.blocks(0.1, 3),
        ])
        self.assertEqual(len(keywords), 2)

    def test_phrase_vad_length_limit_does_not_split_one_long_sound_into_commands(self):
        worker = self.make_worker("phrase")
        keywords, transcripts, chunks = self.run_audio(worker, [
            *self.blocks(0, 8), *self.blocks(0.1, 340),
            *self.blocks(0, 8), *self.blocks(0.1, 10), *self.blocks(0, 20),
        ])
        self.assertEqual([text for _, text in keywords], ["kon", "kon"])
        self.assertEqual(transcripts, ["kon", "kon"])
        self.assertEqual(len(chunks), 2)
        self.assertLessEqual(len(chunks[0].samples), 3.02 * 16_000)

    def test_phrase_started_before_rearm_never_reaches_asr(self):
        worker = self.make_worker("phrase")
        keywords, _, chunks = self.run_audio(worker, [
            *self.blocks(0, 8), *self.blocks(0.1, 4), self.rearm,
            *self.blocks(0.1, 20), *self.blocks(0, 20),
            *self.blocks(0.1, 10), *self.blocks(0, 20),
        ])
        self.assertEqual(len(chunks), 1)
        self.assertEqual([text for _, text in keywords], ["kon"])

    def test_pronunciation_sample_bypasses_ordinary_onset_gate(self):
        for mode in ("phrase", "loud_sound"):
            with self.subTest(mode=mode):
                worker = self.make_worker(mode)
                samples = []
                worker.sample.connect(lambda request, text: samples.append((request, text)))
                with patch("kon.speech.time.monotonic", return_value=100.0):
                    worker.request_sample(17)
                keywords, transcripts, chunks = self.run_audio(worker, [
                    *self.blocks(0.1, 10), *self.blocks(0, 20),
                ])
                self.assertEqual(samples, [(17, "kon")])
                self.assertEqual((keywords, transcripts), ([], []))
                self.assertEqual(len(chunks), 1)

    def test_callback_cannot_enqueue_audio_across_an_asr_or_queue_reset(self):
        worker = self.make_worker()

        def reset_during_callback(*args):
            with worker._lock:
                worker._transcribing = True
                worker._clear_audio()
                worker._transcribing = False
            return 100.1

        with patch("kon.speech.captured_audio_end", side_effect=reset_during_callback):
            worker._microphone_callback(
                np.full((BLOCK_FRAMES, 1), 0.1, dtype=np.float32), BLOCK_FRAMES,
                SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.02), None,
            )
        self.assertTrue(worker._audio.empty())


if __name__ == "__main__":
    unittest.main()
