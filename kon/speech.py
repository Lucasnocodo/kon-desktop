"""Local, gesture-gated microphone recognition using whisper.cpp.

``keyword`` carries the monotonic time at the end of captured audio, never the
time Whisper finished. The UI should check that timestamp against its current
arm time and freshness limit before acting on a file.
"""

from __future__ import annotations

from .paths import resource_root, is_bundled_root, helper_path

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave

import numpy as np
from PySide6.QtCore import QThread, Signal
import sounddevice as sd

from .audio_devices import resolve_microphone


DEFAULT_TRIGGER_SETTING = "kon"
DEFAULT_KON_ALIASES = ("kon", "con")
LOUD_SOUND_MARKER = "大聲短音"
SAMPLE_RATE = 16_000
BLOCK_FRAMES = 320
COMMAND_QUIET_DURATION = 0.16


def normalize_utterance(text: str) -> str:
    """Trim outer punctuation without joining separate words or Chinese words."""
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    while text and (text[0].isspace() or unicodedata.category(text[0]).startswith("P")):
        text = text[1:]
    while text and (text[-1].isspace() or unicodedata.category(text[-1]).startswith("P")):
        text = text[:-1]
    return " ".join(text.split())


def parse_trigger_phrases(setting: str) -> tuple[str, ...]:
    """Validate user aliases and return whole-utterance matches, primary first.

    Only the single default setting ``kon`` adds pronunciation aliases. An
    explicit custom list is matched exactly. Limits apply before expansion.
    """
    if not isinstance(setting, str):
        raise ValueError("語音觸發詞必須是文字。")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs", "Co", "Cn"} and not char.isspace() for char in setting):
        raise ValueError("語音觸發詞包含無效或隱藏字元。")
    entries = re.split(r"[,，、;；]", setting)
    if len(entries) > 8:
        raise ValueError("最多可設定 8 個語音觸發詞，請用逗號分隔。")
    phrases = []
    for entry in entries:
        phrase = normalize_utterance(entry)
        if not phrase:
            raise ValueError("語音觸發詞不可留空，也不能有空白的別名。")
        if len(phrase) > 32:
            raise ValueError("每個語音觸發詞最多 32 個字元。")
        if not any(unicodedata.category(char)[0] in "LN" for char in phrase):
            raise ValueError("語音觸發詞必須包含文字或數字。")
        if any(unicodedata.category(char)[0] not in "LMN" and char not in " '-’" for char in phrase):
            raise ValueError("語音觸發詞只能包含文字、數字、空白、連字號或撇號。")
        phrases.append(phrase)
    if phrases == [DEFAULT_TRIGGER_SETTING]:
        return DEFAULT_KON_ALIASES
    return tuple(dict.fromkeys(phrases))


def matches_keyword(text: str, setting: str = DEFAULT_TRIGGER_SETTING) -> bool:
    """Only complete configured phrases match; never substrings of other words."""
    phrases = parse_trigger_phrases(setting)
    return isinstance(text, str) and normalize_utterance(text) in phrases


def _configured_phrases(setting: str, accepted_phrases: tuple[str, ...] | None) -> tuple[str, ...]:
    """Keep legacy defaults while honoring aliases explicitly chosen in calibration."""
    phrases = parse_trigger_phrases(setting)
    if accepted_phrases is None:
        return phrases
    if not isinstance(accepted_phrases, tuple) or not accepted_phrases or len(accepted_phrases) > 8:
        raise ValueError("校正咒語必須包含 1 至 8 個完整觸發詞。")
    selected = []
    for entry in accepted_phrases:
        # Validate the full entry with the same limits, but do not inherit the
        # implicit default kon/con expansion for a user's explicit selection.
        parse_trigger_phrases(entry)
        if re.search(r"[,，、;；]", entry):
            raise ValueError("每個校正觸發詞必須是完整的一句咒語，不能包含分隔符號。")
        normalized = normalize_utterance(entry)
        if normalized not in phrases:
            raise ValueError("校正觸發詞必須包含在設定的咒語或別名中。")
        selected.append(normalized)
    if phrases[0] not in selected:
        raise ValueError("校正觸發詞必須包含主要咒語。")
    return tuple(dict.fromkeys((phrases[0], *selected)))


def _trigger_language(primary: str, override: str | None) -> str:
    if override is not None:
        if not isinstance(override, str) or not re.fullmatch(r"[a-z]{2,3}|auto", override.strip().casefold()):
            raise ValueError("辨識語言須使用 Whisper 語言代碼，例如 en 或 zh。")
        return override.strip().casefold()
    # The default and Latin phrases use English pronunciation. A phrase with
    # CJK ideographs uses Mandarin; mixed aliases inherit their primary phrase.
    if any("CJK" in unicodedata.name(char, "") for char in primary):
        return "zh"
    return "en"


def captured_audio_end(now: float, adc_start: float, stream_now: float, frames: int) -> float:
    """Translate PortAudio's stream clock into our monotonic clock."""
    duration = frames / SAMPLE_RATE
    lag = stream_now - adc_start - duration
    if not math.isfinite(lag) or lag < 0:
        lag = 0.0
    return now - lag


def whisper_command(
    cli: Path,
    model: Path,
    recording: Path,
    *,
    language: str = "en",
    use_gpu: bool = False,
) -> list[str]:
    # tiny is fast on the CPU and this also works when macOS has no usable Metal
    # device. Some environments abort inside ggml during Metal initialization.
    command = [
        str(cli), "--model", str(model), "--file", str(recording),
        "--language", language, "--no-timestamps", "--no-prints",
        "--no-fallback", "--temperature", "0", "--threads", "4",
    ]
    if not use_gpu:
        command.append("--no-gpu")
    return command


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    captured_end: float


class LoudSoundDetector:
    """Detect one brief loud burst without storing audio or waiting for ASR.

    Three ordinary microphone blocks (60 ms) are sufficient. A burst stays
    latched until 160 ms of quiet, so holding a vowel cannot repeat a command.
    """

    def __init__(
        self,
        energy_threshold: float = 0.025,
        minimum_duration: float = 0.06,
        quiet_duration: float = 0.16,
    ):
        if any(not math.isfinite(value) or value <= 0 for value in (
            energy_threshold, minimum_duration, quiet_duration,
        )):
            raise ValueError("短音的音量與持續時間必須是正數。")
        self.energy_threshold = energy_threshold
        self.minimum_duration = minimum_duration
        self.quiet_duration = quiet_duration
        self.reset()

    def reset(self) -> None:
        self._loud_duration = 0.0
        self._quiet_duration = 0.0
        self._latched = False

    def feed(self, samples: np.ndarray, end_time: float) -> float | None:
        duration = len(samples) / SAMPLE_RATE
        if not duration or not math.isfinite(end_time) or not np.isfinite(samples).all():
            self.reset()
            return None
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if rms < self.energy_threshold:
            self._loud_duration = 0.0
            if self._latched:
                self._quiet_duration += duration
                if self._quiet_duration + 1e-9 >= self.quiet_duration:
                    self.reset()
            return None
        self._quiet_duration = 0.0
        if self._latched:
            return None
        self._loud_duration += duration
        if self._loud_duration + 1e-9 < self.minimum_duration:
            return None
        self._latched = True
        return end_time


class _CommandAudioGate:
    """Require observed quiet before accepting an ordinary command onset.

    Inactive and ASR-busy audio is discarded by the callback, so elapsed wall
    time cannot establish quiet. Only fresh, contiguous microphone samples do.
    Pronunciation recording deliberately bypasses this gate.
    """

    def __init__(self, energy_threshold: float):
        self.energy_threshold = energy_threshold
        self.reset()

    def reset(self) -> None:
        self._quiet_duration = 0.0
        self._last_end: float | None = None
        self._ready = False

    def feed(self, samples: np.ndarray, end_time: float) -> bool:
        duration = len(samples) / SAMPLE_RATE
        if not duration or not math.isfinite(end_time) or not np.isfinite(samples).all():
            self.reset()
            return False
        if self._last_end is not None:
            elapsed = end_time - self._last_end
            # ADC-to-monotonic translation can jitter slightly. A missing,
            # duplicate or reordered block cannot bridge the quiet interval.
            if elapsed <= 0:
                self.reset()
                return False
            if abs(elapsed - duration) > 0.005:
                self.reset()
        self._last_end = end_time
        if self._ready:
            return True
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if rms >= self.energy_threshold:
            self._quiet_duration = 0.0
            return False
        self._quiet_duration += duration
        self._ready = self._quiet_duration + 1e-9 >= COMMAND_QUIET_DURATION
        return self._ready


class SpeechSegmenter:
    """Small bounded energy VAD; stores at most about three seconds of audio."""

    def __init__(self, energy_threshold: float = 0.012):
        self.energy_threshold = energy_threshold
        self.reset()

    def reset(self) -> None:
        self._pre_roll: deque[np.ndarray] = deque(maxlen=8)
        self._frames: list[np.ndarray] = []
        self._noise = self.energy_threshold / 4
        self._above_count = 0
        self._voiced_duration = 0.0
        self._silence_duration = 0.0
        self._total_duration = 0.0

    def feed(self, samples: np.ndarray, end_time: float) -> AudioChunk | None:
        duration = len(samples) / SAMPLE_RATE
        if not duration or not np.isfinite(samples).all():
            return None
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        voiced = rms >= max(self.energy_threshold, self._noise * 3.5)

        if not self._frames:
            self._pre_roll.append(samples.copy())
            self._above_count = self._above_count + 1 if voiced else 0
            if not voiced:
                self._noise = self._noise * 0.98 + rms * 0.02
            if self._above_count >= 2:
                self._frames = list(self._pre_roll)
                self._total_duration = sum(len(frame) for frame in self._frames) / SAMPLE_RATE
                self._voiced_duration = duration * 2
                self._silence_duration = 0.0
            return None

        self._frames.append(samples.copy())
        self._total_duration += duration
        if voiced:
            self._voiced_duration += duration
            self._silence_duration = 0.0
        else:
            self._silence_duration += duration

        if self._silence_duration < 0.32 and self._total_duration < 3.0:
            return None

        chunk = None
        if self._voiced_duration >= 0.12:
            chunk = AudioChunk(np.concatenate(self._frames), end_time)
        self.reset()
        return chunk


class SpeechWorker(QThread):
    keyword = Signal(float, str)
    transcript = Signal(str)
    sample = Signal(int, str)
    sample_failed = Signal(int, str)
    status = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        project_root: str | Path | None = None,
        parent=None,
        *,
        energy_threshold: float = 0.012,
        use_gpu: bool = False,
        trigger_setting: str = DEFAULT_TRIGGER_SETTING,
        trigger_mode: str = "phrase",
        language: str | None = None,
        accepted_phrases: tuple[str, ...] | None = None,
        microphone_setting: dict | str | None = None,
    ):
        super().__init__(parent)
        self.project_root = Path(project_root or resource_root())
        self.cli_path = self.project_root / "vendor/whisper.cpp/build/bin/whisper-cli"
        if is_bundled_root(self.project_root):
            self.cli_path = helper_path("whisper-cli")
        self.model_path = self.project_root / "models/ggml-tiny.bin"
        self.energy_threshold = energy_threshold
        self.microphone_setting = dict(microphone_setting) if isinstance(microphone_setting, dict) else microphone_setting
        self.input_device = None
        self.use_gpu = use_gpu
        self.trigger_mode = self._validate_trigger_mode(trigger_mode)
        phrases = _configured_phrases(trigger_setting, accepted_phrases)
        self._trigger_phrases = frozenset(phrases)
        self.trigger_setting = trigger_setting
        self.trigger_display = phrases[0]
        self.language = _trigger_language(self.trigger_display, language)
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._active = False
        self._sample_request_id: int | None = None
        self._sample_language: str | None = None
        self._armed_since = math.inf
        self._generation = 0
        self._transcribing = False
        self._process: subprocess.Popen | None = None
        self._audio: queue.Queue[tuple[int, np.ndarray, float]] = queue.Queue(maxsize=25)
        self._audio_epoch = 0
        self._last_keyword_audio_end = -math.inf
        self._overflow = threading.Event()

    def set_active(self, active: bool) -> None:
        """Idempotent arm/disarm. Disarming invalidates audio and recognition."""
        with self._lock:
            # Gesture updates must not turn a pronunciation sample into a
            # command. Disarming remains an explicit cancellation of either.
            if active and self._sample_request_id is not None:
                return
            if self._stopping.is_set() or self._active == bool(active):
                return
            self._active = bool(active)
            self._sample_request_id = None
            self._sample_language = None
            self._armed_since = time.monotonic() if active else math.inf
            self._generation += 1
            self._clear_audio()
            process = self._process if not active else None
        self._cancel_process(process)
        self.status.emit(self._armed_status() if active else "等待圈圈手勢")

    @staticmethod
    def _validate_trigger_mode(trigger_mode: str) -> str:
        if trigger_mode not in ("phrase", "loud_sound"):
            raise ValueError("語音觸發方式必須是 phrase 或 loud_sound。")
        return trigger_mode

    def _armed_status(self) -> str:
        with self._lock:
            if self.trigger_mode == "loud_sound":
                return "鎖定已就緒，請稍停後大聲唸一個新短音"
            return f"手勢已就緒，請說 {self.trigger_display}（先稍停再開始）"

    def request_sample(self, request_id: int, *, language: str = "en") -> bool:
        """Capture one utterance for calibration using this worker's microphone.

        Samples never emit ``keyword`` or the ordinary ``transcript`` signal.
        Replacing a sample invalidates its queued audio and in-flight result.
        The caller uses its request ID to reject signals queued before cancel.
        """
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            raise ValueError("校正錄音編號必須是整數。")
        sample_language = _trigger_language(self.trigger_display, language)
        with self._lock:
            if self._stopping.is_set():
                return False
            self._generation += 1
            self._active = True
            self._sample_request_id = request_id
            self._sample_language = sample_language
            self._armed_since = time.monotonic()
            self._clear_audio()
            process = self._process
        self._cancel_process(process)
        self.status.emit("發音校正已就緒，請唸一次咒語，唸完後稍停")
        return True

    def cancel_sample(self) -> None:
        """Cancel a pending sample; a late cancel leaves ordinary arming alone."""
        with self._lock:
            if self._sample_request_id is None:
                return
            self._active = False
            self._sample_request_id = None
            self._sample_language = None
            self._armed_since = math.inf
            self._generation += 1
            self._clear_audio()
            process = self._process
        self._cancel_process(process)
        self.status.emit("發音校正錄音已取消")

    def configure_trigger(
        self,
        trigger_setting: str,
        *,
        trigger_mode: str | None = None,
        language: str | None = None,
        accepted_phrases: tuple[str, ...] | None = None,
    ) -> None:
        """Apply validated phrases while inactive, without reopening the mic."""
        phrases = _configured_phrases(trigger_setting, accepted_phrases)
        trigger_language = _trigger_language(phrases[0], language)
        if trigger_mode is not None:
            self._validate_trigger_mode(trigger_mode)
        with self._lock:
            if self._active:
                raise ValueError("請先停止語音辨識或校正錄音，再修改咒語。")
            self.trigger_setting = trigger_setting
            self.trigger_display = phrases[0]
            self._trigger_phrases = frozenset(phrases)
            self.language = trigger_language
            if trigger_mode is not None:
                self.trigger_mode = trigger_mode
            self._generation += 1
            self._last_keyword_audio_end = -math.inf
            self._clear_audio()
            process = self._process
        self._cancel_process(process)

    def _clear_audio(self) -> None:
        """Drop queued blocks and invalidate callbacks already copying audio."""
        self._audio_epoch += 1
        while True:
            try:
                self._audio.get_nowait()
            except queue.Empty:
                break
        self._overflow.clear()

    def stop(self) -> None:
        """Request bounded shutdown. Create a new worker for the next session."""
        with self._lock:
            self._stopping.set()
            self._active = False
            self._sample_request_id = None
            self._sample_language = None
            self._armed_since = math.inf
            self._generation += 1
            self._clear_audio()
            process = self._process
        self._cancel_process(process)

    @staticmethod
    def _cancel_process(process: subprocess.Popen | None) -> None:
        if process is not None:
            try:
                process.terminate()
            except (ProcessLookupError, OSError):
                pass

    def _current(self, generation: int) -> bool:
        with self._lock:
            return self._active and not self._stopping.is_set() and self._generation == generation

    def _microphone_callback(self, indata, frames, clock, stream_status) -> None:
        with self._lock:
            if stream_status:
                self._overflow.set()
            if not self._active or self._transcribing or self._stopping.is_set():
                return
            generation = self._generation
            armed_since = self._armed_since
            audio_epoch = self._audio_epoch
        try:
            end_time = captured_audio_end(
                time.monotonic(), float(clock.inputBufferAdcTime), float(clock.currentTime), frames
            )
            if end_time - frames / SAMPLE_RATE < armed_since:
                # Hardware can deliver pre-arm audio after the gesture event.
                return
            block = np.asarray(indata[:, 0], dtype=np.float32).copy()
            with self._lock:
                if (
                    not self._current(generation)
                    or self._transcribing
                    or audio_epoch != self._audio_epoch
                ):
                    return
                self._audio.put_nowait((generation, block, end_time))
        except queue.Full:
            self._overflow.set()

    def _publish_result(self, text: str, captured_end: float, generation: int) -> None:
        # Keep the generation check and signal emission together. A subsequent
        # disarm can still race queued UI delivery: the UI also checks arm time.
        with self._lock:
            if not self._current(generation):
                return
            if self._sample_request_id is not None:
                request_id = self._sample_request_id
                self._finish_sample()
                self.sample.emit(request_id, text)
                self.status.emit("發音校正辨識完成" if text else "未辨識出文字，請重新錄製校正咒語")
                return
            if self.trigger_mode != "phrase":
                return
            self.transcript.emit(text)
            if normalize_utterance(text) in self._trigger_phrases and captured_end - self._last_keyword_audio_end >= 1.5:
                self._last_keyword_audio_end = captured_end
                self.keyword.emit(captured_end, normalize_utterance(text))

    def _publish_loud_sound(self, captured_end: float, generation: int) -> None:
        """Publish capture time only while this ordinary loud trigger is armed."""
        with self._lock:
            if (
                not self._current(generation)
                or self._sample_request_id is not None
                or self.trigger_mode != "loud_sound"
            ):
                return
            self.keyword.emit(captured_end, LOUD_SOUND_MARKER)

    def _finish_sample(self) -> None:
        """Disarm a completed sample. Caller holds the state lock."""
        self._active = False
        self._sample_request_id = None
        self._sample_language = None
        self._armed_since = math.inf
        self._generation += 1
        self._clear_audio()

    def _fail_sample(self, message: str, generation: int) -> bool:
        with self._lock:
            if not self._current(generation) or self._sample_request_id is None:
                return False
            request_id = self._sample_request_id
            self._finish_sample()
            self.sample_failed.emit(request_id, message)
            self.status.emit("發音校正辨識失敗，請重新錄製")
            return True

    def _transcribe(self, chunk: AudioChunk, generation: int) -> str | None:
        with tempfile.TemporaryDirectory(prefix="kon-audio-") as temporary:
            recording = Path(temporary) / "utterance.wav"
            pcm = (np.clip(chunk.samples, -1, 1) * 32767).astype("<i2")
            with wave.open(str(recording), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm.tobytes())

            with self._lock:
                if not self._current(generation):
                    return None
                command = whisper_command(
                    self.cli_path, self.model_path, recording,
                    language=self._sample_language or self.language, use_gpu=self.use_gpu,
                )
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._process = process
            deadline = time.monotonic() + 20.0
            try:
                while True:
                    if not self._current(generation):
                        self._cancel_process(process)
                        return None
                    if time.monotonic() >= deadline:
                        raise RuntimeError("離線辨識超過 20 秒，請重啟監聽或檢查 Whisper 安裝。")
                    try:
                        stdout, stderr = process.communicate(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if not self._current(generation):
                    return None
                if process.returncode:
                    detail = stderr.decode("utf-8", errors="replace").strip()[-600:]
                    raise RuntimeError(f"Whisper 離線辨識失敗：{detail or process.returncode}")
                return stdout.decode("utf-8", errors="replace").strip()
            finally:
                if process.poll() is None:
                    self._cancel_process(process)
                    try:
                        process.communicate(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate(timeout=0.5)
                with self._lock:
                    if self._process is process:
                        self._process = None

    def _require_recognizer(self) -> None:
        if not self.cli_path.is_file():
            raise RuntimeError("找不到 Whisper 程式，請先執行 scripts/build-whisper.sh。")
        if not self.model_path.is_file():
            raise RuntimeError("找不到離線模型 models/ggml-tiny.bin，請先完成模型下載。")

    def run(self) -> None:
        segmenter = SpeechSegmenter(self.energy_threshold)
        loud_detector = LoudSoundDetector()
        command_gate = _CommandAudioGate(
            loud_detector.energy_threshold if self.trigger_mode == "loud_sound" else self.energy_threshold
        )
        observed_generation = -1
        try:
            if self._stopping.is_set():
                return
            if self.trigger_mode == "phrase":
                self._require_recognizer()
            device = resolve_microphone(self.microphone_setting)
            self.input_device = device
            if self._stopping.is_set():
                return
            self.status.emit(f"正在開啟麥克風：{device.name}；音訊只在本機辨識")
            with sd.InputStream(
                device=device.index,
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=BLOCK_FRAMES, callback=self._microphone_callback,
            ):
                with self._lock:
                    if self._sample_request_id is not None:
                        ready_status = "發音校正已就緒，請唸一次咒語，唸完後稍停"
                    elif self._active:
                        ready_status = self._armed_status()
                    else:
                        ready_status = "麥克風已就緒，等待圈圈手勢"
                    self.status.emit(f"{device.name} · {ready_status}")
                while not self._stopping.is_set():
                    try:
                        generation, samples, end_time = self._audio.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if not self._current(generation):
                        segmenter.reset()
                        loud_detector.reset()
                        command_gate.reset()
                        continue
                    if generation != observed_generation:
                        segmenter.reset()
                        loud_detector.reset()
                        with self._lock:
                            command_gate.energy_threshold = (
                                loud_detector.energy_threshold
                                if self.trigger_mode == "loud_sound"
                                else self.energy_threshold
                            )
                        command_gate.reset()
                        observed_generation = generation
                    if time.monotonic() - end_time > 0.6:
                        segmenter.reset()
                        loud_detector.reset()
                        command_gate.reset()
                        continue
                    if self._overflow.is_set():
                        segmenter.reset()
                        loud_detector.reset()
                        command_gate.reset()
                        with self._lock:
                            self._clear_audio()
                            retry_status = (
                                "麥克風音訊中斷，請再唸一次校正咒語"
                                if self._sample_request_id is not None
                                else "麥克風音訊中斷，請再大聲唸一個短音"
                                if self.trigger_mode == "loud_sound"
                                else f"麥克風音訊中斷，請再說一次 {self.trigger_display}"
                            )
                            self.status.emit(retry_status)
                        continue
                    with self._lock:
                        sample_mode = self._sample_request_id is not None
                        loud_mode = self.trigger_mode == "loud_sound" and not sample_mode
                    if not sample_mode and not command_gate.feed(samples, end_time):
                        segmenter.reset()
                        loud_detector.reset()
                        continue
                    if loud_mode:
                        captured_end = loud_detector.feed(samples, end_time)
                        if captured_end is not None:
                            self._publish_loud_sound(captured_end, generation)
                        continue
                    chunk = segmenter.feed(samples, end_time)
                    if chunk is None:
                        continue
                    # VAD also ends a segment at its length bound. Audio that
                    # continues through ASR must not become another command.
                    command_gate.reset()
                    with self._lock:
                        if not self._current(generation):
                            continue
                        self._transcribing = True
                        self._clear_audio()
                        sample_mode = self._sample_request_id is not None
                    self.status.emit("正在本機辨識校正發音…" if sample_mode else "正在本機辨識…")
                    try:
                        self._require_recognizer()
                        result = self._transcribe(chunk, generation)
                        if result is not None:
                            self._publish_result(result, chunk.captured_end, generation)
                    except Exception as exc:
                        # A canceled inference is already obsolete. Sample
                        # failures are retryable without opening a second mic.
                        with self._lock:
                            if self._current(generation) and not self._fail_sample(str(exc), generation):
                                raise
                    finally:
                        with self._lock:
                            if self._current(generation):
                                self._clear_audio()
                            self._transcribing = False
                    if self._current(generation):
                        self.status.emit(self._armed_status())
        except sd.PortAudioError as exc:
            if not self._stopping.is_set():
                message = (
                    "無法開啟麥克風。請在 macOS「系統設定 → 隱私權與安全性 → 麥克風」"
                    f"允許啟動 KON 的程式，並確認輸入裝置可用。\n{exc}"
                )
                self._fail_sample(message, self._generation)
                self.error.emit(message)
        except Exception as exc:
            if not self._stopping.is_set():
                self._fail_sample(str(exc), self._generation)
                self.error.emit(str(exc))
        finally:
            self.stop()
            self.status.emit("語音監聽已停止")


def offline_smoke_test(project_root: str | Path | None = None) -> dict[str, object]:
    """Exercise the real multilingual model and CLI without opening a microphone.

    The English repository fixture verifies transcription/output separation.
    Running it with zh additionally checks initialization, not Mandarin accuracy.
    """
    root = Path(project_root or resource_root())
    cli = root / "vendor/whisper.cpp/build/bin/whisper-cli"
    model = root / "models/ggml-tiny.bin"
    recording = root / "vendor/whisper.cpp/samples/jfk.wav"
    for path in (cli, model, recording):
        if not path.is_file():
            raise RuntimeError(f"Missing local smoke-test input: {path}")
    transcripts = {}
    for language in ("en", "zh"):
        result = subprocess.run(
            whisper_command(cli, model, recording, language=language),
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode:
            raise RuntimeError(f"Whisper smoke test failed: {result.stderr[-600:]}")
        text = result.stdout.strip()
        if not text or "read_audio_data:" in text or "whisper_init" in text:
            raise RuntimeError("Whisper did not return clean transcription on stdout.")
        if language == "en" and "fellow americans" not in text.casefold():
            raise RuntimeError(f"Unexpected English fixture transcription: {text}")
        if matches_keyword(text):
            raise RuntimeError("Non-command fixture unexpectedly matched a keyword.")
        transcripts[language] = text
    return {
        "ok": True, "microphone_opened": False, "inference": "local CPU",
        "english_transcript": transcripts["en"],
        "mandarin_initialization": True,
        "mandarin_accuracy_tested": False,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="KON local speech diagnostics")
    parser.add_argument("--self-test", action="store_true", help="Run a local audio fixture; do not open the microphone")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(offline_smoke_test(), ensure_ascii=False, indent=2))
    else:
        parser.print_help()
