"""Language is presentation state, independent of sensors and user phrases."""
import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from kon import i18n
from kon.app import Window
from kon.pronunciation_dialog import PronunciationDialog
from kon.overlay import ScreenOverlay


@pytest.fixture
def application():
    app = QApplication.instance() or QApplication([])
    i18n._catalog.cache_clear()
    i18n.set_language('zh-TW')
    yield app
    i18n.set_language('zh-TW')
    i18n.retranslate_all()


def test_existing_widgets_retranslate_without_changing_item_data(application):
    label = i18n.QLabel('取消')
    item = i18n.QTableWidgetItem('取消')
    combo = i18n.QComboBox()
    combo.addItem('英文', 'en')
    events = []
    combo.currentIndexChanged.connect(events.append)
    i18n.set_language('en')
    i18n.retranslate_all()
    assert label.text() == 'Cancel' and item.text() == 'Cancel'
    assert combo.currentData() == 'en' and not events
    i18n.set_language('zh-TW')
    i18n.retranslate_all()
    assert label.text() == '取消' and item.text() == '取消'


def test_language_preference_preserves_other_user_settings(tmp_path, monkeypatch):
    monkeypatch.delenv('KON_LANGUAGE', raising=False)
    state = tmp_path / '.kon'
    state.mkdir()
    settings = {'voice_trigger': '左上', 'selection_settings': {'window_seconds': 3}}
    (state / 'preferences.json').write_text(json.dumps(settings))
    i18n.save_language('en', tmp_path)
    saved = json.loads((state / 'preferences.json').read_text())
    assert saved == dict(settings, ui_language='en')
    assert i18n.preferred_language(tmp_path) == 'en'
    assert i18n.preferred_language(tmp_path, 'zh-TW') == 'zh-TW'


def test_switching_window_language_keeps_session_and_undo_state(application, tmp_path, monkeypatch):
    monkeypatch.setattr('kon.app.ROOT', tmp_path)
    window = Window(aim_mode='hand')
    receipt = object()
    window.receipt = receipt
    original_gate = window.gate
    window.language_combo.setCurrentIndex(window.language_combo.findData('en'))
    assert i18n.get_language() == 'en'
    assert window.camera is None and window.speech is None
    assert window.receipt is receipt and window.gate is original_gate
    assert window.mode.count() == 2 and not hasattr(window, 'acquisition')
    assert 'Start' in window.start_button.text()
    window.language_combo.setCurrentIndex(window.language_combo.findData('zh-TW'))
    assert window.start_button.text() == '啟動鏡頭與麥克風'
    window.close()


def test_recognition_text_and_filenames_are_never_translated(application):
    i18n.set_language('en')
    assert i18n.translate(i18n.literal('左上')) == '左上'
    editor = i18n.QLineEdit('左上')
    checkbox = i18n.QCheckBox(i18n.literal('左上'))
    assert editor.text() == '左上' and checkbox.text() == '左上'
    metrics = i18n.QFontMetrics(QFont('PingFang TC', 12))
    lines = ScreenOverlay._banner_lines('已鎖定 左上 · 請說 kon', metrics, 900)
    assert '左上' in lines[0]
    i18n.set_language('zh-TW')
    i18n.retranslate_all()
    assert editor.text() == '左上' and checkbox.text() == '左上'


def test_english_interface_does_not_change_internal_diagnostics(application):
    from kon.geometry import classify_gao
    before = classify_gao([])
    i18n.set_language('en')
    after = classify_gao([])
    assert before == after


@pytest.mark.parametrize('language', ['en', 'zh-TW'])
def test_controls_fit_visible_width_after_language_switch(application, tmp_path, monkeypatch, language):
    monkeypatch.setattr('kon.app.ROOT', tmp_path)
    window = Window()
    try:
        window.show()
        window.language_combo.setCurrentIndex(window.language_combo.findData(language))
        window.resize(window.minimumWidth(), window.height())
        application.processEvents()
        scroll = window.controls_scroll
        assert scroll.widget().width() <= scroll.viewport().width()
        assert scroll.horizontalScrollBar().maximum() == 0
    finally:
        window.close()
