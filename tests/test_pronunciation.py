import json

import pytest

from kon.pronunciation import PronunciationProfile, validate_pronunciation_phrase
from kon.speech import matches_keyword


def profile(**changes):
    values = {
        "phrase": "Kon", "language": "en", "samples": ["Kon!", "Con.", "Go"],
        "aliases": ["Kon!", "Con."], "verification": "Con!",
    }
    return PronunciationProfile.create(**(values | changes))


def test_profile_uses_only_explicit_choices_and_normalizes_whole_phrases():
    saved = profile()
    assert saved.phrase == "kon"
    assert saved.samples == ("kon", "con", "go")
    assert saved.aliases == ("kon", "con")
    assert saved.trigger_setting == "kon, con"
    assert saved.verification == "con"
    assert matches_keyword("Con!", saved.trigger_setting)
    assert not matches_keyword("Go", saved.trigger_setting)
    assert not matches_keyword("please con", saved.trigger_setting)


def test_single_phrase_validator_does_not_expand_kon_or_accept_alias_lists():
    assert validate_pronunciation_phrase("Kon!") == "kon"
    assert validate_pronunciation_phrase("召喚狐狸。") == "召喚狐狸"
    with pytest.raises(ValueError):
        validate_pronunciation_phrase("kon, con")


def test_primary_stays_first_and_aliases_are_deduplicated():
    saved = profile(phrase="Summon fox", samples=["Fox", "Summon fox", "Fox!"],
                    aliases=["Fox!", "Fox", "Summon fox"], verification="Summon fox!")
    assert saved.aliases == ("fox", "summon fox")
    assert saved.trigger_setting == "summon fox, fox"


def test_chinese_phrase_is_saved_and_matched_as_a_complete_utterance():
    saved = profile(phrase="召喚狐狸", language="zh", samples=["召喚狐狸。", "招喚狐狸", "召喚狐狸"],
                    aliases=["召喚狐狸", "招喚狐狸"], verification="招喚狐狸。")
    assert saved.trigger_setting == "召喚狐狸, 招喚狐狸"
    assert matches_keyword("招喚狐狸。", saved.trigger_setting)
    assert not matches_keyword("我要招喚狐狸", saved.trigger_setting)


@pytest.mark.parametrize("changes", [
    {"phrase": ""}, {"phrase": "!!!"}, {"phrase": "a" * 33}, {"phrase": "go, fox"},
    {"phrase": "go\u200b"}, {"language": "auto"}, {"language": None},
    {"samples": ["kon", "con"]}, {"samples": ["kon", "con", ""]},
    {"samples": ["kon", "con", "go; fox"]}, {"samples": "kon"},
    {"samples": None}, {"aliases": []}, {"aliases": ["fox"]},
    {"aliases": "kon"}, {"verification": "go"}, {"verification": "please con"},
])
def test_invalid_or_unselected_enrollment_is_rejected(changes):
    with pytest.raises(ValueError):
        profile(**changes)


def test_builtin_con_alias_does_not_bypass_explicit_verification():
    with pytest.raises(ValueError):
        profile(aliases=["kon"], verification="con")


def test_atomic_store_preserves_camera_and_other_preferences(tmp_path):
    path = tmp_path / ".kon/preferences.json"
    path.parent.mkdir()
    previous = {"camera_settings": {"camera": 2, "eye": "left", "fov": 80.25}, "other": [1, 2]}
    path.write_text(json.dumps(previous))
    saved = profile()
    assert saved.save(tmp_path) == path
    data = json.loads(path.read_text())
    assert data["camera_settings"] == previous["camera_settings"]
    assert data["other"] == previous["other"]
    assert data["voice_trigger"] == "kon, con"
    assert data["voice_language"] == "en"
    assert data["voice_pronunciation"]["version"] == 1
    assert PronunciationProfile.load(tmp_path) == saved
    assert not list(path.parent.glob("*.tmp"))
    assert [item.name for item in path.parent.iterdir()] == ["preferences.json"]


@pytest.mark.parametrize("field, value", [("voice_trigger", "go"), ("voice_language", "zh")])
def test_manual_trigger_or_language_edits_invalidate_saved_profile(tmp_path, field, value):
    path = profile().save(tmp_path)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PronunciationProfile.load(tmp_path)


@pytest.mark.parametrize("payload", [None, {}, {"version": 2}, {"version": True},
                                      {"version": 1}, {"version": 1, "phrase": "kon"}])
def test_missing_or_invalid_profiles_raise_validation_error(tmp_path, payload):
    path = tmp_path / ".kon/preferences.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"voice_pronunciation": payload}))
    with pytest.raises(ValueError):
        PronunciationProfile.load(tmp_path)


def test_replace_failure_preserves_previous_store_and_cleans_temporary(tmp_path, monkeypatch):
    path = profile().save(tmp_path)
    before = path.read_bytes()

    def fail(*args):
        raise OSError("cannot replace")

    monkeypatch.setattr("kon.pronunciation.os.replace", fail)
    with pytest.raises(OSError, match="cannot replace"):
        profile(verification="kon").save(tmp_path)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("contents", ["invalid json", "null", "[]"])
def test_invalid_existing_preferences_are_not_overwritten(tmp_path, contents):
    path = tmp_path / ".kon/preferences.json"
    path.parent.mkdir()
    path.write_text(contents)
    with pytest.raises(ValueError):
        profile().save(tmp_path)
    assert path.read_text() == contents


def test_unvalidated_direct_constructor_cannot_save(tmp_path):
    saved = PronunciationProfile("kon", "en", ("kon", "con", "go"), ("fox",), "fox")
    with pytest.raises(ValueError):
        saved.save(tmp_path)
    assert not (tmp_path / ".kon/preferences.json").exists()
