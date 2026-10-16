"""Validated pronunciation enrollment, without retaining microphone audio."""

from __future__ import annotations

from .paths import state_root as user_state_dir

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

from .speech import normalize_utterance, parse_trigger_phrases


def _single_phrase(value: str, label: str) -> str:
    if not isinstance(value, str) or re.search(r"[,，、;；]", value):
        raise ValueError(f"{label}須是一句短咒語，不能包含別名分隔符號。")
    # The backend validates length and characters. A single default `kon`
    # expands to its built-in aliases, but enrollment records only this phrase.
    parse_trigger_phrases(value)
    return normalize_utterance(value)


def validate_pronunciation_phrase(text: str) -> str:
    """Validate one enrollment phrase without expanding backend aliases."""
    return _single_phrase(text, "咒語")


def _phrases(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label}須是咒語清單。")
    try:
        return tuple(_single_phrase(value, label) for value in values)
    except TypeError as exc:
        raise ValueError(f"{label}須是咒語清單。") from exc


@dataclass(frozen=True)
class PronunciationProfile:
    phrase: str
    language: str
    samples: tuple[str, ...]
    aliases: tuple[str, ...]
    verification: str

    @classmethod
    def create(
        cls,
        phrase: str,
        language: str,
        samples: Sequence[str],
        aliases: Sequence[str],
        verification: str,
    ) -> PronunciationProfile:
        phrase = _single_phrase(phrase, "咒語")
        if language not in ("en", "zh"):
            raise ValueError("發音校正語言須選擇英文或中文。")
        samples = _phrases(samples, "錄音辨識結果")
        if len(samples) != 3:
            raise ValueError("請完成三次咒語錄音後再驗證。")
        aliases = tuple(dict.fromkeys(_phrases(aliases, "選取的發音")))
        if not aliases:
            raise ValueError("請至少選取一個錄音辨識結果作為可接受的發音。")
        if any(alias not in samples for alias in aliases):
            raise ValueError("可接受的發音必須從這三次錄音的辨識結果中選取。")
        accepted = tuple(dict.fromkeys((phrase, *aliases)))
        parse_trigger_phrases(", ".join(accepted))
        verification = _single_phrase(verification, "驗證辨識結果")
        # Verify explicit enrollment choices rather than default `kon`'s
        # implicit `con` expansion or other unselected recording results.
        if verification not in accepted:
            raise ValueError("驗證結果不符合咒語或你選取的發音，請重新錄音。")
        return cls(phrase, language, samples, aliases, verification)

    @property
    def trigger_setting(self) -> str:
        return ", ".join(dict.fromkeys((self.phrase, *self.aliases)))

    def save(self, root: str | Path) -> Path:
        """Atomically replace only pronunciation preferences in the store."""
        profile = self.create(self.phrase, self.language, self.samples, self.aliases, self.verification)
        destination = user_state_dir(Path(root)) / 'preferences.json'
        try:
            preferences = json.loads(destination.read_text(encoding="utf-8"))
        except FileNotFoundError:
            preferences = {}
        if not isinstance(preferences, dict):
            raise ValueError("偏好設定格式無效，無法保存發音校正。")
        preferences.update(
            voice_trigger=profile.trigger_setting,
            voice_language=profile.language,
            voice_pronunciation={
                "version": 1,
                "phrase": profile.phrase,
                "language": profile.language,
                "samples": list(profile.samples),
                "aliases": list(profile.aliases),
                "verification": profile.verification,
            },
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent,
                prefix="preferences-", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(preferences, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, root: str | Path) -> PronunciationProfile:
        preferences = json.loads((user_state_dir(Path(root)) / 'preferences.json').read_text(encoding="utf-8"))
        if not isinstance(preferences, dict):
            raise ValueError("偏好設定格式無效。")
        payload = preferences.get("voice_pronunciation")
        if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 1:
            raise ValueError("找不到有效的發音校正資料。")
        try:
            profile = cls.create(
                payload["phrase"], payload["language"], payload["samples"],
                payload["aliases"], payload["verification"],
            )
        except KeyError as exc:
            raise ValueError("發音校正資料不完整。") from exc
        if (preferences.get("voice_trigger") != profile.trigger_setting
                or preferences.get("voice_language") != profile.language):
            raise ValueError("咒語或語言已修改，請重新進行發音校正。")
        return profile
