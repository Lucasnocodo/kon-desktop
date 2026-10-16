import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kon.speech import (
    AudioChunk,
    BLOCK_FRAMES,
    SAMPLE_RATE,
    SpeechSegmenter,
    SpeechWorker,
    captured_audio_end,
    matches_keyword,
    parse_trigger_phrases,
)


class KeywordTests(unittest.TestCase):
    def test_isolated_keyword_with_punctuation(self):
        for text in ("kon", "KON!", "ＫＯＮ。", "con", "（CON）"):
            with self.subTest(text=text):
                self.assertTrue(matches_keyword(text))

    def test_substrings_and_other_audio_cannot_trigger(self):
        for text in ("go", "goh", "gow", "gou", "gao", "狗", "夠", "够", "勾", "溝", "沟", "購", "购", "no", "kon away", "我說kon", "con artist", "konkon", "kon kon", "k.o.n", "[BLANK_AUDIO]", "", "garbage"):
            with self.subTest(text=text):
                self.assertFalse(matches_keyword(text))

    def test_default_parser_keeps_primary_first_and_expands_aliases(self):
        phrases = parse_trigger_phrases(" ＫＯＮ！ ")
        self.assertEqual(phrases, ("kon", "con"))

    def test_custom_aliases_and_phrase_whitespace(self):
        self.assertEqual(parse_trigger_phrases(" Kon,ＣＯＮ； 吃   掉、狗 "), ("kon", "con", "吃 掉", "狗"))
        self.assertEqual(parse_trigger_phrases("co\u0301n, cón"), ("cón",))
        self.assertTrue(matches_keyword("（CON）", "kon, con"))
        self.assertTrue(matches_keyword("  eat \t it。 ", "eat it"))
        self.assertTrue(matches_keyword("吃掉！", "吃掉"))
        self.assertEqual(parse_trigger_phrases("go"), ("go",))
        self.assertTrue(matches_keyword("GO!", "go"))
        self.assertFalse(matches_keyword("狗", "go"))
        self.assertFalse(matches_keyword("go", "kon, con"))
        self.assertFalse(matches_keyword("狗", "go, kon"))
        self.assertFalse(matches_keyword("我說吃掉", "吃掉"))
        self.assertFalse(matches_keyword("吃", "吃掉"))

    def test_invalid_settings_are_rejected_before_starting_sensors(self):
        invalid = (None, 4, [], "", "  ", ",", "go,", "go,,kon", "go；；kon", "a" * 33, "1,2,3,4,5,6,7,8,9", "go\0", "g\u200bo", "👻", "go/google")
        for setting in invalid:
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                parse_trigger_phrases(setting)
        self.assertEqual(len(parse_trigger_phrases(",".join(str(index) for index in range(8)))), 8)

    def test_worker_uses_configured_keywords_display_and_language(self):
        worker = SpeechWorker(trigger_setting=" Kon , con ")
        self.assertEqual(worker.trigger_setting, " Kon , con ")
        self.assertEqual(worker.trigger_display, "kon")
        self.assertEqual(worker.language, "en")
        self.assertEqual(SpeechWorker().trigger_display, "kon")
        self.assertEqual(SpeechWorker().language, "en")
        self.assertEqual(SpeechWorker(trigger_setting="吃掉").language, "zh")
        self.assertEqual(SpeechWorker(trigger_setting="go", language="zh").language, "zh")
        events, statuses = [], []
        worker.keyword.connect(lambda stamp, text: events.append((stamp, text)))
        worker.status.connect(statuses.append)
        worker.set_active(True)
        self.assertIn("請說 kon", statuses[-1])
        worker._publish_result("go", 10, worker._generation)
        worker._publish_result("con", 11, worker._generation)
        self.assertEqual(events, [(11, "con")])

    def test_explicit_calibrated_phrases_do_not_accept_unselected_default_alias(self):
        worker = SpeechWorker(accepted_phrases=("KON!",))
        keywords = []
        worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
        worker.set_active(True)
        worker._publish_result("con", 10.0, worker._generation)
        worker._publish_result("kon", 11.0, worker._generation)
        self.assertEqual(keywords, [(11.0, "kon")])
        worker.set_active(False)
        worker.configure_trigger("kon", accepted_phrases=("con", "kon"))
        self.assertEqual(worker.trigger_display, "kon")
        worker.set_active(True)
        worker._publish_result("con", 1.0, worker._generation)
        self.assertEqual(keywords, [(11.0, "kon"), (1.0, "con")])
        worker.set_active(False)
        worker.configure_trigger("kon", accepted_phrases=("kon",))
        worker.set_active(True)
        worker._publish_result("con", 3.0, worker._generation)
        self.assertEqual(len(keywords), 2)

    def test_invalid_explicit_phrases_cannot_broaden_the_setting(self):
        for selected in ((), ["kon"], ("con",), ("kon", "dog"), ("kon, con",), ("kon", ""), ("kon", 3)):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                SpeechWorker(accepted_phrases=selected)
        worker = SpeechWorker(trigger_setting="吃掉, 發動", accepted_phrases=("吃掉",))
        for selected in (("發動",), ("吃掉", "kon")):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                worker.configure_trigger("吃掉, 發動", accepted_phrases=selected)
            self.assertEqual(worker._trigger_phrases, frozenset(("吃掉",)))

    def test_capture_timestamp_removes_input_device_latency(self):
        # A 20 ms block began at stream time 5.0, callback at 5.1:
        # its last sample is 80 ms old, even if inference happens much later.
        self.assertAlmostEqual(captured_audio_end(100.0, 5.0, 5.1, 320), 99.92)

    def test_stale_generation_is_discarded_after_rearming(self):
        worker = SpeechWorker()
        events = []
        worker.keyword.connect(lambda stamp, text: events.append((stamp, text)))
        worker.set_active(True)
        old_generation = worker._generation
        worker.set_active(False)
        worker.set_active(True)
        worker._publish_result("kon", 50.0, old_generation)
        self.assertEqual(events, [])
        worker._publish_result("kon", 60.0, worker._generation)
        self.assertEqual(events, [(60.0, "kon")])

    def test_keyword_timestamp_and_cooldown_use_capture_time(self):
        worker = SpeechWorker()
        events = []
        worker.keyword.connect(lambda stamp, text: events.append((stamp, text)))
        worker.set_active(True)
        worker._publish_result("kon", 10.0, worker._generation)
        worker._publish_result("con", 10.5, worker._generation)
        worker._publish_result("con", 11.5, worker._generation)
        self.assertEqual(events, [(10.0, "kon"), (11.5, "con")])
        worker.stop()
        worker._publish_result("kon", 20.0, worker._generation)
        self.assertEqual(len(events), 2)

    def test_delayed_hardware_audio_from_before_arm_is_discarded(self):
        worker = SpeechWorker()
        with patch("kon.speech.time.monotonic", return_value=100.0):
            worker.set_active(True)
        block = np.ones((BLOCK_FRAMES, 1), dtype=np.float32) * 0.1
        with patch("kon.speech.time.monotonic", return_value=100.1):
            worker._microphone_callback(
                block, BLOCK_FRAMES,
                SimpleNamespace(inputBufferAdcTime=5.0, currentTime=5.2), None,
            )
        self.assertTrue(worker._audio.empty())
        with patch("kon.speech.time.monotonic", return_value=100.2):
            worker._microphone_callback(
                block, BLOCK_FRAMES,
                SimpleNamespace(inputBufferAdcTime=5.0, currentTime=5.02), None,
            )
        self.assertFalse(worker._audio.empty())


class SegmenterTests(unittest.TestCase):
    def test_silence_and_impulse_do_not_become_speech(self):
        segmenter = SpeechSegmenter()
        silence = np.zeros(BLOCK_FRAMES, dtype=np.float32)
        for index in range(100):
            self.assertIsNone(segmenter.feed(silence, index * 0.02))
        self.assertIsNone(segmenter.feed(np.full(BLOCK_FRAMES, 0.1, dtype=np.float32), 2.02))
        for index in range(30):
            self.assertIsNone(segmenter.feed(silence, 2.04 + index * 0.02))

    def test_speech_ends_on_silence_with_original_audio_end(self):
        segmenter = SpeechSegmenter()
        chunk = None
        for index in range(20):
            chunk = segmenter.feed(np.full(BLOCK_FRAMES, 0.1, dtype=np.float32), 100.0 + index * 0.02)
            self.assertIsNone(chunk)
        for index in range(20):
            audio_end = 100.4 + index * 0.02
            chunk = segmenter.feed(np.zeros(BLOCK_FRAMES, dtype=np.float32), audio_end)
            if chunk is not None:
                break
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.captured_end, audio_end)
        self.assertLessEqual(len(chunk.samples), SAMPLE_RATE * 3.1)

    def test_continuous_audio_is_bounded(self):
        segmenter = SpeechSegmenter()
        chunk = None
        for index in range(200):
            chunk = segmenter.feed(np.full(BLOCK_FRAMES, 0.1, dtype=np.float32), index * 0.02)
            if chunk is not None:
                break
        self.assertIsNotNone(chunk)
        self.assertLessEqual(len(chunk.samples), SAMPLE_RATE * 3.05)

    def test_reset_discards_unfinished_previous_arm_audio(self):
        segmenter = SpeechSegmenter()
        for index in range(10):
            segmenter.feed(np.full(BLOCK_FRAMES, 0.1, dtype=np.float32), index * 0.02)
        segmenter.reset()
        for index in range(25):
            self.assertIsNone(segmenter.feed(np.zeros(BLOCK_FRAMES, dtype=np.float32), 10 + index * 0.02))


class SampleCaptureTests(unittest.TestCase):
    def test_sample_is_single_shot_and_never_emits_a_command(self):
        worker = SpeechWorker()
        samples, keywords, transcripts = [], [], []
        worker.sample.connect(lambda request, text: samples.append((request, text)))
        worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
        worker.transcript.connect(transcripts.append)
        worker.set_active(True)
        ordinary_generation = worker._generation
        self.assertTrue(worker.request_sample(0, language="zh"))
        generation = worker._generation
        worker.set_active(True)  # A gesture cannot switch capture mode.
        worker._publish_result("kon", 10.0, ordinary_generation)
        worker._publish_result("kon", 11.0, generation)
        worker._publish_result("kon", 12.0, generation)
        self.assertEqual(samples, [(0, "kon")])
        self.assertEqual(keywords, [])
        self.assertEqual(transcripts, [])
        self.assertFalse(worker._active)
        self.assertIsNone(worker._sample_request_id)
        self.assertEqual(worker.language, "en")
        self.assertFalse(worker._current(generation))

    def test_cancel_and_replacement_discard_old_samples_and_queued_audio(self):
        worker = SpeechWorker()
        samples = []
        worker.sample.connect(lambda request, text: samples.append((request, text)))
        worker.request_sample(1)
        first_generation = worker._generation
        worker._audio.put((first_generation, np.zeros(BLOCK_FRAMES, dtype=np.float32), 1.0))
        worker._overflow.set()
        worker.request_sample(2)
        second_generation = worker._generation
        self.assertTrue(worker._audio.empty())
        self.assertFalse(worker._overflow.is_set())
        worker._publish_result("kon", 10.0, first_generation)
        worker.cancel_sample()
        worker._publish_result("kon", 11.0, second_generation)
        self.assertEqual(samples, [])
        self.assertFalse(worker._active)
        worker.set_active(True)
        ordinary_generation = worker._generation
        worker.cancel_sample()  # A late dialog cleanup preserves new arming.
        self.assertTrue(worker._current(ordinary_generation))

    def test_disarming_cancels_a_sample_and_stopped_workers_refuse_new_samples(self):
        worker = SpeechWorker()
        samples = []
        worker.sample.connect(lambda request, text: samples.append((request, text)))
        worker.request_sample(1)
        generation = worker._generation
        worker.set_active(False)
        worker._publish_result("kon", 10.0, generation)
        self.assertEqual(samples, [])
        worker.stop()
        self.assertFalse(worker.request_sample(2))
        self.assertFalse(worker._active)

    def test_empty_sample_completes_so_the_ui_can_offer_a_retry(self):
        worker = SpeechWorker()
        samples = []
        worker.sample.connect(lambda request, text: samples.append((request, text)))
        worker.request_sample(3)
        worker._publish_result("", 10.0, worker._generation)
        self.assertEqual(samples, [(3, "")])
        self.assertFalse(worker._active)
        self.assertTrue(worker.request_sample(4))

    def test_invalid_sample_arguments_do_not_replace_an_active_request(self):
        worker = SpeechWorker()
        worker.request_sample(1)
        generation = worker._generation
        for request_id, language in ((True, "en"), ("1", "en"), (2, "english")):
            with self.subTest(request_id=request_id, language=language), self.assertRaises(ValueError):
                worker.request_sample(request_id, language=language)
            self.assertTrue(worker._current(generation))
            self.assertEqual(worker._sample_request_id, 1)

    def test_reconfiguration_validates_then_resets_language_queue_and_cooldown(self):
        worker = SpeechWorker()
        keywords = []
        worker.keyword.connect(lambda stamp, text: keywords.append((stamp, text)))
        worker.set_active(True)
        old_generation = worker._generation
        worker._publish_result("kon", 100.0, old_generation)
        with self.assertRaises(ValueError):
            worker.configure_trigger("吃掉")
        worker.set_active(False)
        worker._audio.put((old_generation, np.zeros(BLOCK_FRAMES, dtype=np.float32), 1.0))
        worker.configure_trigger("吃掉, 發動")
        self.assertEqual(worker.trigger_display, "吃掉")
        self.assertEqual(worker.language, "zh")
        self.assertTrue(worker._audio.empty())
        worker.set_active(True)
        worker._publish_result("kon", 101.0, old_generation)
        worker._publish_result("kon", 102.0, worker._generation)
        worker._publish_result("發動", 1.0, worker._generation)
        self.assertEqual(keywords, [(100.0, "kon"), (1.0, "發動")])
        worker.set_active(False)
        for setting, language in (("go,", "en"), ("go", "English")):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                worker.configure_trigger(setting, language=language)
            self.assertEqual(worker.trigger_display, "吃掉")
            self.assertEqual(worker.language, "zh")
        worker.configure_trigger("go", language="auto")
        self.assertEqual(worker.language, "auto")
        worker.request_sample(7)
        with self.assertRaises(ValueError):
            worker.configure_trigger("kon")

    def test_sample_failure_can_retry_without_reopening_the_microphone(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = SpeechWorker(temporary)
            worker.cli_path = Path(temporary) / "whisper"
            worker.model_path = Path(temporary) / "model.bin"
            worker.cli_path.touch()
            worker.model_path.touch()
            samples, failures, fatal_errors, languages = [], [], [], []

            def queue_block():
                worker._audio.put((worker._generation, np.ones(BLOCK_FRAMES, dtype=np.float32), time.monotonic()))

            def retry(request_id, message):
                failures.append((request_id, message))
                worker.request_sample(2, language="zh")
                queue_block()

            def complete(request_id, text):
                samples.append((request_id, text))
                worker.stop()

            def transcribe(chunk, generation):
                languages.append(worker._sample_language)
                if len(languages) == 1:
                    raise RuntimeError("temporary recognition failure")
                return "kon"

            worker.sample_failed.connect(retry)
            worker.sample.connect(complete)
            worker.error.connect(fatal_errors.append)
            worker.request_sample(1, language="en")
            with patch("kon.speech.resolve_microphone", return_value=SimpleNamespace(index=0, name="Test mic")), patch("kon.speech.sd.InputStream") as microphone, patch("kon.speech.SpeechSegmenter") as segmenter, patch.object(worker, "_transcribe", side_effect=transcribe):
                microphone.return_value.__enter__.side_effect = queue_block
                segmenter.return_value.feed.side_effect = lambda samples, stamp: AudioChunk(samples, stamp)
                worker.run()
            microphone.assert_called_once()
            self.assertEqual(failures, [(1, "temporary recognition failure")])
            self.assertEqual(samples, [(2, "kon")])
            self.assertEqual(languages, ["en", "zh"])
            self.assertEqual(fatal_errors, [])


class RecognitionProcessTests(unittest.TestCase):
    def make_worker(self, temporary: str, program: str, **settings) -> SpeechWorker:
        directory = Path(temporary)
        executable = directory / "fake-whisper"
        executable.write_text("#!/bin/sh\n" + program)
        executable.chmod(0o700)
        model = directory / "model.bin"
        model.touch()
        worker = SpeechWorker(directory, **settings)
        worker.cli_path = executable
        worker.model_path = model
        worker.set_active(True)
        return worker

    def test_local_process_preserves_capture_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = self.make_worker(temporary, "printf 'kon.\\n'\n")
            chunk = AudioChunk(np.zeros(1600, dtype=np.float32), 1.25)
            generation = worker._generation
            result = worker._transcribe(chunk, generation)
            self.assertEqual(result, "kon.")
            events = []
            worker.keyword.connect(lambda stamp, text: events.append((stamp, text)))
            worker._publish_result(result, chunk.captured_end, generation)
            self.assertEqual(events, [(1.25, "kon")])
            self.assertIsNone(worker._process)

    def test_subprocess_receives_inferred_and_overridden_language(self):
        program = 'while [ "$#" -gt 0 ]; do\nif [ "$1" = "--language" ]; then\nshift\nprintf "%s" "$1"\nexit 0\nfi\nshift\ndone\nexit 1\n'
        for settings, expected in (({}, "en"), ({"trigger_setting": "吃掉"}, "zh"), ({"trigger_setting": "go", "language": "zh"}, "zh")):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as temporary:
                worker = self.make_worker(temporary, program, **settings)
                text = worker._transcribe(AudioChunk(np.zeros(1600, dtype=np.float32), 1), worker._generation)
                self.assertEqual(text, expected)

    def test_sample_language_is_used_only_for_that_capture(self):
        program = 'while [ "$#" -gt 0 ]; do\nif [ "$1" = "--language" ]; then\nshift\nprintf "%s" "$1"\nexit 0\nfi\nshift\ndone\nexit 1\n'
        with tempfile.TemporaryDirectory() as temporary:
            worker = self.make_worker(temporary, program)
            worker.request_sample(4, language="zh")
            generation = worker._generation
            chunk = AudioChunk(np.zeros(1600, dtype=np.float32), 1.0)
            text = worker._transcribe(chunk, generation)
            self.assertEqual(text, "zh")
            worker._publish_result(text, chunk.captured_end, generation)
            self.assertEqual(worker.language, "en")
            worker.set_active(True)
            self.assertEqual(worker._transcribe(chunk, worker._generation), "en")

    def test_cli_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = self.make_worker(temporary, "echo 'model load failed' >&2\nexit 3\n")
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                worker._transcribe(AudioChunk(np.zeros(1600, dtype=np.float32), 1.0), worker._generation)
            self.assertIsNone(worker._process)

    def test_disarming_cancels_subprocess_and_discards_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            # exec means termination reaches the actual process rather than a
            # shell parent leaving a child holding the stdout/stderr pipes open.
            worker = self.make_worker(temporary, "exec sleep 10\n")
            result = []
            thread = threading.Thread(target=lambda: result.append(
                worker._transcribe(AudioChunk(np.zeros(1600, dtype=np.float32), 1.0), worker._generation)
            ))
            thread.start()
            deadline = time.monotonic() + 2
            while worker._process is None and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(worker._process)
            worker.set_active(False)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [None])
            self.assertIsNone(worker._process)

    def test_canceling_sample_terminates_inference_and_discards_its_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = self.make_worker(temporary, "exec sleep 10\n")
            worker.request_sample(10)
            generation = worker._generation
            results, samples = [], []
            worker.sample.connect(lambda request_id, text: samples.append((request_id, text)))
            thread = threading.Thread(target=lambda: results.append(
                worker._transcribe(AudioChunk(np.zeros(1600, dtype=np.float32), 1.0), generation)
            ))
            thread.start()
            deadline = time.monotonic() + 2
            while worker._process is None and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(worker._process)
            worker.cancel_sample()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results, [None])
            worker._publish_result("kon", 10.0, generation)
            self.assertEqual(samples, [])
            self.assertIsNone(worker._process)


if __name__ == "__main__":
    unittest.main()
