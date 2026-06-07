from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.config.config import save_config
from src.effects.effect import param_specs
from src.effects.effect_manager import EffectManager
from src.effects.registry import EFFECT_REGISTRY
from src.gestures.gesture_manager import GestureManager, GestureType
from src.gestures.registry import GESTURE_REGISTRY

CONFIG_PATH = "src/config/app.json"


class ControlPanel(QWidget):
    def __init__(
        self, gesture_manager: GestureManager, effect_manager: EffectManager, fps_counter=None
    ):
        super().__init__()
        self.gesture_manager = gesture_manager
        self.effect_manager = effect_manager
        self.fps_counter = fps_counter

        name_by_class = {cls: name for name, cls in GESTURE_REGISTRY.items()}
        self.detectors_by_name = {
            name_by_class[type(d)]: d for d in gesture_manager.detectors if type(d) in name_by_class
        }
        self._row_containers = {}

        layout = QVBoxLayout(self)
        for gesture_name in self.detectors_by_name:
            layout.addWidget(self._gesture_group(gesture_name))
        layout.addWidget(self._gesture_params_group())
        save = QPushButton("Save to app.json")
        save.clicked.connect(self._save)
        layout.addWidget(save)
        layout.addWidget(self._state_group())
        layout.addStretch()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_state)
        self._timer.start(100)

    # ---------- bindings editor ----------
    def _gesture_group(self, gesture_name):
        box = QGroupBox(gesture_name)
        outer = QVBoxLayout(box)
        rows = QVBoxLayout()
        self._row_containers[gesture_name] = rows
        outer.addLayout(rows)
        add = QPushButton("+ Add effect")
        add.clicked.connect(lambda _=False, g=gesture_name: self._add_effect(g))
        outer.addWidget(add)
        self._rebuild_rows(gesture_name)
        return box

    def _pool(self, gesture_name):
        return list(self.effect_manager.bindings.get(GestureType(gesture_name), []))

    def _commit(self, gesture_name, pool):
        self.effect_manager.set_pool(GestureType(gesture_name), pool)
        self._rebuild_rows(gesture_name)

    def _rebuild_rows(self, gesture_name):
        rows = self._row_containers[gesture_name]
        while rows.count():
            w = rows.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        pool = self._pool(gesture_name)
        for i, effect in enumerate(pool):
            rows.addWidget(self._effect_row(gesture_name, i, effect, len(pool)))

    def _effect_row(self, gesture_name, index, effect, pool_len):
        row = QWidget()
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        combo = QComboBox()
        combo.addItems(list(EFFECT_REGISTRY.keys()))
        combo.setCurrentText(effect.name)
        combo.currentTextChanged.connect(
            lambda new, g=gesture_name, i=index: self._change_effect(g, i, new)
        )
        header.addWidget(combo, 1)
        for text, slot, enabled in (
            ("\u2191", lambda _=False, g=gesture_name, i=index: self._move(g, i, -1), index > 0),
            (
                "\u2193",
                lambda _=False, g=gesture_name, i=index: self._move(g, i, +1),
                index < pool_len - 1,
            ),
            ("\u2715", lambda _=False, g=gesture_name, i=index: self._remove(g, i), True),
        ):
            btn = QToolButton()
            btn.setText(text)
            btn.setEnabled(enabled)
            btn.clicked.connect(slot)
            header.addWidget(btn)
        v.addLayout(header)
        v.addLayout(self._param_form(effect))
        return row

    def _param_form(self, effect):
        form = QFormLayout()
        for name, typ, default in param_specs(type(effect)):
            value = getattr(effect, name, default)
            if typ is bool:
                w = QCheckBox()
                w.setChecked(bool(value))
                w.toggled.connect(lambda v, e=effect, n=name: setattr(e, n, v))
            elif typ is int:
                w = QSpinBox()
                w.setRange(0, 1000)
                w.setValue(int(value))
                w.valueChanged.connect(lambda v, e=effect, n=name: setattr(e, n, v))
            elif typ is str:
                w = QComboBox()
                choices = list(getattr(type(effect), "_COLORS", {}).keys()) or [str(value)]
                w.addItems(choices)
                w.setCurrentText(str(value))
                w.currentTextChanged.connect(lambda v, e=effect, n=name: setattr(e, n, v))
            else:
                continue  # skip unknown param types instead of crashing
            form.addRow(name, w)
        return form

    def _default_effect(self):
        return EFFECT_REGISTRY[next(iter(EFFECT_REGISTRY))]()

    def _add_effect(self, g):
        pool = self._pool(g)
        pool.append(self._default_effect())
        self._commit(g, pool)

    def _remove(self, g, i):
        pool = self._pool(g)
        if 0 <= i < len(pool):
            pool.pop(i)
        self._commit(g, pool)

    def _move(self, g, i, d):
        pool = self._pool(g)
        j = i + d
        if 0 <= i < len(pool) and 0 <= j < len(pool):
            pool[i], pool[j] = pool[j], pool[i]
        self._commit(g, pool)

    def _change_effect(self, g, i, new_name):
        pool = self._pool(g)
        if 0 <= i < len(pool) and pool[i].name != new_name:
            pool[i] = EFFECT_REGISTRY[new_name]()
            self._commit(g, pool)

    def _gesture_params_group(self):
        box = QGroupBox("Gesture tuning")
        outer = QVBoxLayout(box)
        for name, det in self.detectors_by_name.items():
            specs = param_specs(type(det))
            if not specs:
                continue
            sub = QGroupBox(name)
            form = QFormLayout(sub)
            for pname, typ, default in specs:
                if typ is float:
                    w = QDoubleSpinBox()
                    w.setRange(0.0, 5.0)
                    w.setDecimals(3)
                    w.setSingleStep(0.005)
                    w.setValue(float(getattr(det, pname, default)))
                    w.valueChanged.connect(lambda v, d=det, a=pname: setattr(d, a, v))
                elif typ is int:
                    w = QSpinBox()
                    w.setRange(0, 1000)
                    w.setValue(int(getattr(det, pname, default)))
                    w.valueChanged.connect(lambda v, d=det, a=pname: setattr(d, a, v))
                else:
                    continue
                form.addRow(pname, w)
            outer.addWidget(sub)
        return box

    # ---------- live state ----------
    def _state_group(self):
        box = QGroupBox("Live state")
        v = QVBoxLayout(box)
        self._state_view = QPlainTextEdit()
        self._state_view.setReadOnly(True)
        self._state_view.setMaximumHeight(140)
        v.addWidget(self._state_view)
        return box

    def _refresh_state(self):
        lines = []
        if self.fps_counter is not None:
            lines.append(f"FPS: {self.fps_counter.fps:4.1f}")
        snap = self.effect_manager.snapshot()
        for name, det in self.detectors_by_name.items():
            active = snap.get(name)
            if active:
                lines.append(
                    f"{name}: ACTIVE x{active['regions']} -> {', '.join(active['effects']) or '-'}"
                )
            else:
                lines.append(
                    f"{name}: idle" + (" (armed)" if getattr(det, "active", False) else "")
                )
        self._state_view.setPlainText("\n".join(lines))

    def _save(self):
        save_config(CONFIG_PATH, self.effect_manager, self.detectors_by_name)
