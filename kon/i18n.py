"""Presentation-only localization; tracking state and user data stay unchanged.

Qt display subclasses retain original text so language changes can update the
existing window without restarting sensors or discarding an undo receipt.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
from string import Formatter
import weakref

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QLocale

from .paths import resource_root, state_root

_language = 'zh-TW'
_bindings = weakref.WeakKeyDictionary()
_widgets = weakref.WeakSet()
_HAN = re.compile(r'[\u3400-\u9fff]')


class RawText(str):
    """User-provided data, never a translation key."""


def literal(text):
    return RawText(str(text))


def normalize_language(value):
    if not isinstance(value, str):
        return None
    if value.lower().replace('_', '-').startswith('zh'):
        return 'zh-TW'
    if value.lower().replace('_', '-').startswith('en'):
        return 'en'
    return None


def get_language():
    return _language


def set_language(value):
    global _language
    normalized = normalize_language(value)
    if normalized is None:
        raise ValueError('Supported interface languages: en, zh-TW')
    _language = normalized


def preferred_language(project_root=None, override=None):
    chosen = normalize_language(override or os.environ.get('KON_LANGUAGE'))
    if chosen:
        return chosen
    try:
        saved = json.loads((state_root(project_root) / 'preferences.json').read_text(encoding='utf-8'))
        chosen = normalize_language(saved.get('ui_language'))
        if chosen:
            return chosen
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return normalize_language(QLocale.system().name()) or 'en'


def save_language(value, project_root=None):
    normalized = normalize_language(value)
    if normalized is None:
        raise ValueError('Supported interface languages: en, zh-TW')
    path = state_root(project_root) / 'preferences.json'
    try:
        current = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(current, dict):
            current = {}
    except (OSError, ValueError):
        current = {}
    current['ui_language'] = normalized
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.language.tmp')
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@lru_cache(maxsize=1)
def _catalog():
    directory = resource_root() / 'kon/locales'
    if not directory.is_dir():
        directory = Path(__file__).resolve().parent / 'locales'
    try:
        entries = json.loads((directory / 'en.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        entries = {}
    try:
        diagnostic_fields = json.loads((directory / 'diagnostic-fields.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        diagnostic_fields = {}
    patterns = []
    for source, target in entries.items():
        if not isinstance(source, str) or not isinstance(target, str) or '{' not in source:
            continue
        try:
            parts, fields = [], []
            for text, field, _spec, _conversion in Formatter().parse(source):
                parts.append(re.escape(text))
                if field is not None:
                    fields.append(field)
                    parts.append('(.*?)')
            if fields and any(char not in '{}0123456789 ' for char in source):
                patterns.append((re.compile(''.join(parts), re.DOTALL), fields, target,
                                 set(diagnostic_fields.get(source, []))))
        except ValueError:
            continue
    # Prefer a specific full sentence over a short fragment with a large field.
    patterns.sort(key=lambda value: len(value[0].pattern), reverse=True)
    return entries, patterns


def translate(text, *, _depth=0):
    if isinstance(text, RawText) or not isinstance(text, str) or _language != 'en' or not _HAN.search(text):
        return str(text) if isinstance(text, RawText) else text
    entries, patterns = _catalog()
    if text in entries:
        return entries[text]
    if '；' in text and _depth < 4:
        parts = text.split('；')
        localized = [translate(part, _depth=_depth + 1) for part in parts]
        if localized != parts and all(not _HAN.search(part) for part in localized):
            return '; '.join(localized)
    for pattern, fields, target, diagnostics in patterns:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        values = {field: match.group(index + 1) for index, field in enumerate(fields)}
        if _depth < 4:
            for field in diagnostics:
                field = str(field)
                if field in values:
                    values[field] = translate(values[field], _depth=_depth + 1)
        try:
            return target.format(*(values.get(str(index), '') for index in range(len(fields))))
        except (ValueError, IndexError, KeyError):
            continue
    # Composed status messages often join complete independent instructions.
    if '\n' in text:
        return '\n'.join(translate(line, _depth=_depth + 1) for line in text.split('\n'))
    return text


def _remember(obj, setter, source):
    if isinstance(source, RawText):
        if obj in _bindings:
            _bindings[obj].pop(setter, None)
        return
    try:
        _bindings.setdefault(obj, {})[setter] = source
    except TypeError:
        pass


def retranslate_all():
    for obj, properties in list(_bindings.items()):
        try:
            for setter, source in properties.items():
                if isinstance(obj, _Localized):
                    getattr(super(_Localized, obj), setter)(translate(source))
                else:
                    getattr(obj, setter)(translate(source))
        except RuntimeError:
            continue  # A Qt parent may already have destroyed the C++ object.
    for widget in list(_widgets):
        try:
            callback = getattr(widget, '_translate_items', None)
            if callback:
                callback()
            if hasattr(widget, 'update'):
                widget.update()
        except RuntimeError:
            continue
    app = QtWidgets.QApplication.instance()
    if app:
        for widget in app.topLevelWidgets():
            widget.update()


class _Localized:
    _constructor_setter = None

    def __init__(self, *args, **kwargs):
        source = args[0] if args and isinstance(args[0], str) else None
        if source is not None and self._constructor_setter:
            args = (translate(source), *args[1:])
        super().__init__(*args, **kwargs)
        _widgets.add(self)
        if source is not None and self._constructor_setter:
            _remember(self, self._constructor_setter, source)

    def _translated_set(self, setter, text):
        _remember(self, setter, text)
        getattr(super(_Localized, self), setter)(translate(text))

    def setText(self, text): self._translated_set('setText', text)
    def setWindowTitle(self, text): self._translated_set('setWindowTitle', text)
    def setToolTip(self, text): self._translated_set('setToolTip', text)
    def setAccessibleName(self, text): self._translated_set('setAccessibleName', text)
    def setPlaceholderText(self, text): self._translated_set('setPlaceholderText', text)
    def setPrefix(self, text): self._translated_set('setPrefix', text)
    def setSuffix(self, text): self._translated_set('setSuffix', text)
    def setTitle(self, text): self._translated_set('setTitle', text)


class QLabel(_Localized, QtWidgets.QLabel): _constructor_setter = 'setText'
class QPushButton(_Localized, QtWidgets.QPushButton): _constructor_setter = 'setText'
class QCheckBox(_Localized, QtWidgets.QCheckBox): _constructor_setter = 'setText'
class QDialog(_Localized, QtWidgets.QDialog): pass
class QMainWindow(_Localized, QtWidgets.QMainWindow): pass
class QWidget(_Localized, QtWidgets.QWidget): pass
class QFrame(_Localized, QtWidgets.QFrame): pass
class QSpinBox(_Localized, QtWidgets.QSpinBox): pass
class QDoubleSpinBox(_Localized, QtWidgets.QDoubleSpinBox): pass
class QSystemTrayIcon(_Localized, QtWidgets.QSystemTrayIcon): pass


class QLineEdit(_Localized, QtWidgets.QLineEdit):
    def setText(self, text):
        # Editable text is always user data (phrases, paths, numeric inputs).
        QtWidgets.QLineEdit.setText(self, str(text))


class QComboBox(_Localized, QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        self._source_items = {}
        super().__init__(*args, **kwargs)

    def addItem(self, *args, **kwargs):
        index = 0 if args and isinstance(args[0], str) else 1
        values = list(args)
        if len(values) > index:
            self._source_items[self.count()] = values[index]
            values[index] = translate(values[index])
        return super().addItem(*values, **kwargs)

    def addItems(self, texts):
        for text in texts:
            self.addItem(text)

    def setItemText(self, index, text):
        self._source_items[index] = text
        super().setItemText(index, translate(text))

    def clear(self):
        self._source_items.clear()
        super().clear()

    def _translate_items(self):
        previous = self.blockSignals(True)
        try:
            for index, source in self._source_items.items():
                if index < self.count():
                    super().setItemText(index, translate(source))
        finally:
            self.blockSignals(previous)


class QMenu(_Localized, QtWidgets.QMenu):
    _constructor_setter = 'setTitle'

    def addAction(self, *args, **kwargs):
        values = list(args)
        index = 0 if values and isinstance(values[0], str) else 1
        source = values[index] if len(values) > index and isinstance(values[index], str) else None
        if source is not None:
            values[index] = translate(source)
        action = super().addAction(*values, **kwargs)
        if source is not None:
            _remember(action, 'setText', source)
        return action


class QTableWidgetItem(_Localized, QtWidgets.QTableWidgetItem):
    _constructor_setter = 'setText'
    __hash__ = object.__hash__


class QTableWidget(_Localized, QtWidgets.QTableWidget):
    def setHorizontalHeaderLabels(self, labels):
        self._source_headers = tuple(labels)
        super().setHorizontalHeaderLabels([translate(text) for text in labels])

    def _translate_items(self):
        if hasattr(self, '_source_headers'):
            super().setHorizontalHeaderLabels([translate(text) for text in self._source_headers])


class QTabWidget(_Localized, QtWidgets.QTabWidget):
    def addTab(self, widget, *args):
        values = list(args)
        if values and isinstance(values[-1], str):
            source = values[-1]
            values[-1] = translate(source)
            index = super().addTab(widget, *values)
            if not hasattr(self, '_source_tabs'):
                self._source_tabs = {}
            self._source_tabs[index] = source
            return index
        return super().addTab(widget, *values)

    def _translate_items(self):
        for index, source in getattr(self, '_source_tabs', {}).items():
            self.setTabText(index, translate(source))


class QPainter(QtGui.QPainter):
    def drawText(self, *args):
        return super().drawText(*(translate(value) if isinstance(value, str) else value for value in args))


class QFontMetrics(QtGui.QFontMetrics):
    def boundingRect(self, *args):
        return super().boundingRect(*(translate(value) if isinstance(value, str) else value for value in args))

    def horizontalAdvance(self, text, *args):
        return super().horizontalAdvance(translate(text), *args)

    def elidedText(self, text, *args):
        return super().elidedText(translate(text), *args)
