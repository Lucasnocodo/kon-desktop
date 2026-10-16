"""Microphone routing at the capture boundary without opening audio hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kon.speech import SpeechWorker


MICROPHONE = {"name": "EMEET C960", "hostapi": "Core Audio"}


@pytest.mark.parametrize("mode", ["phrase", "loud_sound"])
@pytest.mark.parametrize("setting", [None, MICROPHONE])
def test_capture_resolves_microphone_before_opening_explicit_device(monkeypatch, mode, setting):
    worker = SpeechWorker(trigger_mode=mode, microphone_setting=setting)
    resolved = SimpleNamespace(index=7, name="EMEET C960")
    resolve = Mock(return_value=resolved)
    stream = Mock()
    # Stop at entry, so the real worker opens/closes its context but never
    # waits for samples or invokes the recognizer.
    stream.return_value.__enter__ = Mock(side_effect=worker.stop)
    stream.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("kon.speech.resolve_microphone", resolve)
    monkeypatch.setattr("kon.speech.sd.InputStream", stream)
    monkeypatch.setattr(worker, "_require_recognizer", lambda: None)
    errors = []
    worker.error.connect(errors.append)

    worker.run()

    resolve.assert_called_once_with(setting)
    stream.assert_called_once()
    assert stream.call_args.kwargs["device"] == resolved.index
    assert errors == []


@pytest.mark.parametrize("mode", ["phrase", "loud_sound"])
@pytest.mark.parametrize("setting", [None, MICROPHONE])
def test_unavailable_microphone_never_opens_default_or_iphone(monkeypatch, mode, setting):
    worker = SpeechWorker(trigger_mode=mode, microphone_setting=setting)
    resolve = Mock(side_effect=RuntimeError("找不到可用的非 iPhone 麥克風"))
    stream = Mock()
    monkeypatch.setattr("kon.speech.resolve_microphone", resolve)
    monkeypatch.setattr("kon.speech.sd.InputStream", stream)
    monkeypatch.setattr(worker, "_require_recognizer", lambda: None)
    errors = []
    worker.error.connect(errors.append)

    worker.run()

    stream.assert_not_called()
    assert len(errors) == 1 and "麥克風" in errors[0]


def test_pronunciation_device_failure_ends_sample_without_opening_audio(monkeypatch):
    worker = SpeechWorker(microphone_setting=MICROPHONE)
    monkeypatch.setattr("kon.speech.resolve_microphone", Mock(side_effect=RuntimeError("已選麥克風已拔除")))
    stream = Mock()
    monkeypatch.setattr("kon.speech.sd.InputStream", stream)
    monkeypatch.setattr(worker, "_require_recognizer", lambda: None)
    failures = []
    worker.sample_failed.connect(lambda request, message: failures.append((request, message)))
    assert worker.request_sample(9, language="en")

    worker.run()

    stream.assert_not_called()
    assert failures == [(9, "已選麥克風已拔除")]
