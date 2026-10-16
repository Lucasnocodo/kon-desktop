"""Offline diagnostics retain transcription checks on slower CPU-only hosts."""
from types import SimpleNamespace

import pytest

from kon import speech


def prepare_inputs(root):
    for name in ('vendor/whisper.cpp/build/bin/whisper-cli', 'models/ggml-tiny.bin',
                 'vendor/whisper.cpp/samples/jfk.wav'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_mandarin_initialization_is_bounded_without_weakening_english_check(tmp_path, monkeypatch):
    prepare_inputs(tmp_path)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        assert kwargs['timeout'] == 120
        assert '--no-gpu' in command
        language = command[command.index('--language') + 1]
        return SimpleNamespace(returncode=0, stderr='', stdout=(
            'My fellow Americans, ask what you can do for your country.' if language == 'en' else ''))
    monkeypatch.setattr(speech.subprocess, 'run', run)
    report = speech.offline_smoke_test(tmp_path)
    assert report['ok'] and report['mandarin_initialization']
    assert not report['microphone_opened'] and not report['mandarin_accuracy_tested']
    assert '--duration' not in commands[0]
    assert commands[1][commands[1].index('--duration') + 1] == '1000'
    assert commands[1][commands[1].index('--beam-size') + 1] == '1'


@pytest.mark.parametrize('stdout', ['', 'unrelated words', 'whisper_init: debug output'])
def test_english_transcription_still_must_be_clean_and_correct(tmp_path, monkeypatch, stdout):
    prepare_inputs(tmp_path)
    monkeypatch.setattr(speech.subprocess, 'run', lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stderr='', stdout=stdout))
    with pytest.raises(RuntimeError):
        speech.offline_smoke_test(tmp_path)


def test_failed_mandarin_initialization_is_not_accepted(tmp_path, monkeypatch):
    prepare_inputs(tmp_path)
    def run(command, **_kwargs):
        language = command[command.index('--language') + 1]
        return SimpleNamespace(returncode=(1 if language == 'zh' else 0), stderr='initialization failed',
                               stdout='My fellow Americans')
    monkeypatch.setattr(speech.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='initialization failed'):
        speech.offline_smoke_test(tmp_path)


def test_mandarin_stdout_diagnostics_are_still_rejected(tmp_path, monkeypatch):
    prepare_inputs(tmp_path)
    def run(command, **_kwargs):
        language = command[command.index('--language') + 1]
        return SimpleNamespace(returncode=0, stderr='', stdout=(
            'whisper_init: unexpected diagnostics' if language == 'zh' else 'My fellow Americans'))
    monkeypatch.setattr(speech.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='clean transcription'):
        speech.offline_smoke_test(tmp_path)
