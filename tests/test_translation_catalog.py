"""English diagnostics compose correctly without translating personal data."""
from __future__ import annotations

import json
from pathlib import Path
import re
from string import Formatter

import pytest

from kon import i18n

LOCALES = Path(__file__).resolve().parents[1] / "kon" / "locales"
HAN = re.compile(r"[\u3400-\u9fff]")


@pytest.fixture(autouse=True)
def english_catalog():
    i18n._catalog.cache_clear()
    i18n.set_language("en")
    yield
    i18n.set_language("zh-TW")
    i18n._catalog.cache_clear()


def test_joined_finger_diagnostics_translate_each_failure():
    first = "食指關節追蹤不完整，請讓手指清楚入鏡"
    second = "小指關節追蹤不完整，請讓手指清楚入鏡"
    result = i18n.translate(first + "；" + second)
    assert result == i18n.translate(first) + "; " + i18n.translate(second)
    assert "index finger" in result and "little finger" in result
    assert not HAN.search(result)


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "角度 1 / 5：中間  ·  中間 → 上 → 下 → 左 → 右",
            "Angle 1 / 5: Center  ·  Center → Up → Down → Left → Right",
        ),
        ("中間角度，已完成", "Center angle, Complete"),
        (
            "正在記錄「中間」角度的新影格，請維持同一手形與角度約一秒。",
            "Recording fresh frames for the “Center” angle. Hold the same hand shape and angle for about one second.",
        ),
        (
            "請使用校正時的左手做出召喚手勢",
            "Make the summoning gesture with the left hand used during calibration",
        ),
        (
            "食指與校正姿勢不同，請重做你的召喚手勢",
            "Your index finger differs from the calibrated pose. Repeat your summoning gesture.",
        ),
        (
            "中指、無名指未靠到拇指指腹，請壓在拇指上",
            "Your middle and ring fingers must touch the thumb pad. Press against your thumb.",
        ),
        ("KON · 增加手勢角度", "KON · Add a gesture angle"),
        ("瞄準校正 2 / 9", "Aiming calibration 2 / 9"),
        ("左上 (18%, 18%)", "Top left (18%, 18%)"),
    ],
)
def test_nested_calibration_labels_are_english(source, expected):
    assert i18n.translate(source) == expected


@pytest.mark.parametrize("side, english_side", [("左", "left"), ("右", "right")])
def test_eye_diagnostic_translates_side_and_preserves_measurement(side, english_side):
    result = i18n.translate(f"{side}眼虹膜太小（4.2 px）；請靠近鏡頭並改善光線")
    assert result.lower() == (
        f"the {english_side} iris is too small (4.2 px). move closer and improve the lighting."
    )
    assert not HAN.search(result)


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "已取得追蹤點；請保持眼睛與手孔位置片刻，等待深度穩定（追蹤約 12.1 FPS，請多停留片刻）",
            "Tracking points acquired. Keep your eye and hand ring still briefly while depth stabilizes. "
            "(Tracking about 12.1 FPS. Hold your aim a little longer.)",
        ),
        (
            "已取得追蹤點；請保持眼睛位置片刻，等待深度穩定（追蹤間隔過長，約 4.2 FPS；請減少其他程式負載）",
            "Tracking points acquired. Keep your eye still briefly while depth stabilizes. "
            "(Tracking interval is too long, about 4.2 FPS. Reduce the load from other apps.)",
        ),
        (
            "鏡頭實際影像 1280×720（比例 1.78:1）；推論 1280×720，保留完整畫面；非 16:9，視角可能因鏡頭裁切而不同",
            "Camera image 1280×720 (aspect 1.78:1); inference 1280×720, preserving the full frame; "
            "not 16:9, so the field of view may differ due to camera cropping",
        ),
        (
            "中位點偏差符合參考值；這次尚未檢查連續影格的抖動。 "
            "精度檢查是選用項目，結果不會清除已保存的九點；可繼續操作已校正範圍內的桌面圖示，或重新記錄選取點改善瞄準。",
            "Median-point error meets the reference threshold. Jitter across consecutive frames has not been checked yet. "
            "The accuracy check is optional and will not erase the saved nine points. Continue using desktop icons "
            "within the calibrated area, or record a selected point again to improve your aim.",
        ),
    ],
)
def test_composed_stream_and_accuracy_statuses_translate_as_one_message(source, expected):
    assert i18n.translate(source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "已鎖定 取消\n追蹤暫時中斷，恢復後即可繼續",
            "Locked: 取消\nTracking interrupted briefly. Continue when it recovers.",
        ),
        ("狐狸吞掉了 中央.txt（演練）", "The fox swallowed 中央.txt (practice)"),
        ("聽到：取消", "Heard: 取消"),
        ("使用辨識結果：左上", "Use transcription: 左上"),
        ("手勢已就緒，請說 吃掉（先稍停再開始）", "Gesture ready. Say 吃掉 after a brief pause."),
        (
            "中央 · 麥克風已就緒，等待圈圈手勢",
            "中央 · Microphone ready. Waiting for the hand ring gesture.",
        ),
        (
            "正在開啟麥克風：取消；音訊只在本機辨識",
            "Opening microphone: 取消. Audio recognition stays on this Mac.",
        ),
    ],
)
def test_user_names_transcripts_and_phrases_remain_opaque(source, expected):
    # These personal values deliberately coincide with existing translation keys.
    assert i18n.translate(source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("聽到：（未辨識）", "Heard: (not recognized)"),
        (
            "這次辨識為「未辨識到咒語」，尚未通過。請再試一次，或重新記錄。",
            "Recognized “No trigger phrase recognized”; the test has not passed yet. Try again or make new recordings.",
        ),
    ],
)
def test_authored_empty_recognition_status_is_translated(source, expected):
    assert i18n.translate(source) == expected


def test_catalog_and_diagnostic_metadata_preserve_placeholder_contracts():
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f"Duplicate catalog key: {key}"
            result[key] = value
        return result

    def fields(text):
        return [field for _, field, _, _ in Formatter().parse(text) if field is not None]

    catalog = json.loads((LOCALES / "en.json").read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    metadata = json.loads((LOCALES / "diagnostic-fields.json").read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    assert catalog and metadata
    for source, target in catalog.items():
        assert isinstance(source, str) and isinstance(target, str)
        assert sorted(fields(source)) == sorted(fields(target)), source
        assert all(field.isdecimal() for field in fields(source)), source
    for source, indexes in metadata.items():
        assert source in catalog, source
        assert len(indexes) == len(set(indexes)), source
        assert set(indexes) <= set(fields(source)), source
    # Nested diagnostics are translatable, while personal values remain opaque.
    assert metadata["已鎖定 {0}\n{1}"] == ["1"]
    assert "0" not in metadata.get("狐狸吞掉了 {0}（演練）", [])
    assert "0" not in metadata.get("聽到：{0}", [])
    assert "0" not in metadata.get("正在開啟麥克風：{0}；音訊只在本機辨識", [])
