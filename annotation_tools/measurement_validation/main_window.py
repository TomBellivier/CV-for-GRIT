"""Main window: navigation between the annotations, measurement classification,
presets, undo/redo and export of the status CSV."""
import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QToolBar, QDockWidget, QPushButton, QInputDialog, QMessageBox, QStatusBar
)

import csv_io
from app_logging import logger
from canvas import MeasurementCanvas
from models import Preset, MEASURABLE, NON_MEASURABLE, edge_key

UNDO_LIMIT = 50
APP_DIR = Path(__file__).resolve().parent
PRESETS_PATH = APP_DIR / "presets.json"
STATE_DIR = APP_DIR / "state"


def _edge_str(key):
    return "::".join(key)


def _edge_from_str(s):
    a, b = s.split("::", 1)
    return edge_key(a, b)


class MainWindow(QMainWindow):
    def __init__(self, config, annotations, project_name, export_dir):
        super().__init__()
        self.config = config
        self.annotations = annotations
        self.project_name = project_name
        self.export_dir = Path(export_dir)
        self.csv_path = self.export_dir / f"{project_name}_measurements.csv"
        self.state_path = STATE_DIR / f"{project_name}_state.json"

        self.index = 0
        self.presets = {}  # name -> Preset
        self.load_presets()

        # snapshots (edge_overrides, done) of the current image
        self.undo_stack = []
        self.redo_stack = []

        self.start_time = time.time()  # timer start, as soon as the CSV is loaded

        self.setWindowTitle("Measurement classification")
        self.resize(1400, 900)

        self.canvas = MeasurementCanvas(config, self)
        self.setCentralWidget(self.canvas)

        self._build_sidebar()
        self._build_toolbar()
        self._build_statusbar()
        self.canvas.edgeChanged.connect(self.set_edge_status)
        self.canvas.strokeStarted.connect(self.push_undo)
        self._build_shortcuts()

    # ---------------- UI construction ----------------
    def _build_sidebar(self):
        dock = QDockWidget("Measurements", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.measure_list = QListWidget()
        self.measure_list.itemClicked.connect(self._on_measure_list_clicked)
        layout.addWidget(QLabel("Measurements (click = toggle measurable / non measurable)"))
        layout.addWidget(self.measure_list)

        layout.addWidget(QLabel("Presets"))
        self.preset_list = QListWidget()
        layout.addWidget(self.preset_list)
        btn_row = QHBoxLayout()
        for label, slot in (("Save preset", self.save_current_as_preset),
                            ("Apply", self.apply_selected_preset),
                            ("Delete", self.delete_selected_preset)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            btn_row.addWidget(button)
        layout.addLayout(btn_row)

        dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.refresh_preset_list()

    def _build_toolbar(self):
        tb = QToolBar("Main")
        self.addToolBar(tb)
        tb.addAction("<< Previous image", self.prev_image)
        tb.addAction("Next image >>", self.next_image)
        tb.addSeparator()
        tb.addAction("Fit view", self.canvas.fit_to_view)
        tb.addAction("Rotate ↺ (Ctrl+Left)", lambda: self.canvas.rotate_view(-90))
        tb.addAction("Rotate ↻ (Ctrl+Right)", lambda: self.canvas.rotate_view(90))
        tb.addSeparator()
        tb.addAction("All measurable", lambda: self.set_all_statuses(MEASURABLE))
        tb.addAction("All non measurable", lambda: self.set_all_statuses(NON_MEASURABLE))
        tb.addSeparator()
        tb.addAction("Undo (Ctrl+Z)", self.undo)
        tb.addAction("Redo (Ctrl+Y)", self.redo)
        tb.addSeparator()
        tb.addAction("Save (Ctrl+S)", self.save)
        tb.addAction("Validate (Enter)", self.mark_done)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.pos_label = QLabel("")
        sb.addWidget(self.pos_label)
        self.timer_label = QLabel("")
        sb.addPermanentWidget(self.timer_label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_timer_display)
        self._timer.start(1000)
        self._update_timer_display()

    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_image)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence(Qt.Key_Return), self, activated=self.mark_done)
        QShortcut(QKeySequence(Qt.Key_Enter), self, activated=self.mark_done)
        QShortcut(QKeySequence("Ctrl+Left"), self, activated=lambda: self.canvas.rotate_view(-90))
        QShortcut(QKeySequence("Ctrl+Right"), self, activated=lambda: self.canvas.rotate_view(90))

    def _update_timer_display(self):
        elapsed = time.time() - self.start_time
        hours, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"

        done_count = sum(1 for a in self.annotations if a.done)
        speed_str = f"{(elapsed / 60.0) / done_count:.1f} min/img" if done_count else "— min/img"
        self.timer_label.setText(f"⏱ {time_str}    {done_count}/{len(self.annotations)} validated    {speed_str}")

    # ---------------- current annotation ----------------
    def current_ann(self):
        return self.annotations[self.index]

    def load_image(self, idx):
        if not self.annotations:
            return
        if not (0 <= idx < len(self.annotations)):
            logger.error("load_image: index %d out of range (%d annotations)", idx, len(self.annotations))
            return
        self.index = idx
        self.undo_stack = []
        self.redo_stack = []
        ann = self.current_ann()
        logger.debug("load_image(%d): %s", idx, ann.image_path)
        self.canvas.show_annotation(ann)
        self.setWindowTitle(f"{ann.image_name}  [{idx + 1}/{len(self.annotations)}]")
        self.pos_label.setText(
            f"Image {idx + 1}/{len(self.annotations)} — {ann.image_name}"
            + ("  ✓ validated" if ann.done else "")
        )
        self.refresh_measure_list()

    def next_image(self):
        if self.index + 1 < len(self.annotations):
            self.load_image(self.index + 1)
        else:
            self.set_status("Last image")

    def prev_image(self):
        if self.index > 0:
            self.load_image(self.index - 1)
        else:
            self.set_status("First image")

    def first_unvalidated_index(self):
        for i, ann in enumerate(self.annotations):
            if not ann.done:
                return i
        return 0

    def set_status(self, msg):
        self.statusBar().showMessage(msg, 4000)

    # ---------------- undo / redo ----------------
    def push_undo(self):
        if not self.annotations:
            return
        ann = self.current_ann()
        self.undo_stack.append((dict(ann.edge_overrides), ann.done))
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _swap_state(self, from_stack, to_stack, label):
        if not from_stack:
            self.set_status(f"Nothing to {label.lower()}")
            return
        ann = self.current_ann()
        to_stack.append((dict(ann.edge_overrides), ann.done))
        ann.edge_overrides, ann.done = from_stack.pop()
        self.canvas.refresh_statuses(ann)
        self.refresh_measure_list()
        self.set_status(label)

    def undo(self):
        self._swap_state(self.undo_stack, self.redo_stack, "Undo")

    def redo(self):
        self._swap_state(self.redo_stack, self.undo_stack, "Redo")

    # ---------------- classification ----------------
    def set_edge_status(self, key, status):
        """Called by the canvas: classify an edge, and with it every edge that must
        follow the same status (see AppConfig.cascade_edges)."""
        ann = self.current_ann()
        for e in self.config.cascade_edges(key):
            a, b = e
            if status == MEASURABLE and not (ann.has_kp(a) and ann.has_kp(b)):
                continue  # an edge without its two kp stays non measurable, override useless
            ann.edge_overrides[e] = status
            self.canvas.set_link_status(e, status)
        self.refresh_measure_list()

    def set_all_statuses(self, status):
        if not self.annotations:
            return
        self.push_undo()
        ann = self.current_ann()
        for key in self.config.edges:
            a, b = key
            if status == MEASURABLE and not (ann.has_kp(a) and ann.has_kp(b)):
                continue
            ann.edge_overrides[key] = status
        self.canvas.refresh_statuses(ann)
        self.refresh_measure_list()

    def refresh_measure_list(self):
        if not self.annotations:
            return
        ann = self.current_ann()
        self.measure_list.clear()
        for name, keys in self.config.measurement_edges.items():
            status = ann.measurement_status(keys)
            label = f"{name}: {'measurable' if status == MEASURABLE else 'non measurable'}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, name)
            item.setBackground(QColor(210, 255, 210) if status == MEASURABLE else QColor(230, 230, 230))
            self.measure_list.addItem(item)

    def _on_measure_list_clicked(self, item):
        name = item.data(Qt.UserRole)
        ann = self.current_ann()
        keys = self.config.measurement_edges[name]
        new_status = NON_MEASURABLE if ann.measurement_status(keys) == MEASURABLE else MEASURABLE
        self.push_undo()
        for key in keys:
            a, b = key
            if new_status == MEASURABLE and not (ann.has_kp(a) and ann.has_kp(b)):
                continue
            ann.edge_overrides[key] = new_status
        self.canvas.refresh_statuses(ann)
        self.refresh_measure_list()

    # ---------------- presets ----------------
    def load_presets(self):
        if not PRESETS_PATH.exists():
            return
        try:
            data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
            self.presets = {n: Preset(n, ov) for n, ov in data.items()}
        except Exception:
            logger.exception("Cannot read the presets: %s", PRESETS_PATH)
            self.presets = {}

    def save_presets(self):
        data = {p.name: p.overrides for p in self.presets.values()}
        PRESETS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def refresh_preset_list(self):
        self.preset_list.clear()
        for name in self.presets:
            self.preset_list.addItem(name)

    def save_current_as_preset(self):
        if not self.annotations:
            return
        name, ok = QInputDialog.getText(self, "New preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        ann = self.current_ann()
        overrides = {_edge_str(key): ann.edge_status(key) for key in self.config.edges}
        self.presets[name] = Preset(name, overrides)
        self.save_presets()
        self.refresh_preset_list()
        self.set_status(f"Preset '{name}' saved")

    def apply_selected_preset(self):
        item = self.preset_list.currentItem()
        if item is None or not self.annotations:
            return
        preset = self.presets.get(item.text())
        if preset is None:
            return
        self.push_undo()
        ann = self.current_ann()
        ann.edge_overrides = {}
        skipped = 0
        for key_str, status in preset.overrides.items():
            key = _edge_from_str(key_str)
            a, b = key
            if status == MEASURABLE and not (ann.has_kp(a) and ann.has_kp(b)):
                skipped += 1
                continue
            ann.edge_overrides[key] = status
        self.canvas.refresh_statuses(ann)
        self.refresh_measure_list()
        msg = f"Preset '{preset.name}' applied"
        if skipped:
            msg += f" ({skipped} segment(s) skipped: missing keypoints)"
        self.set_status(msg)

    def delete_selected_preset(self):
        item = self.preset_list.currentItem()
        if item is None:
            return
        self.presets.pop(item.text(), None)
        self.save_presets()
        self.refresh_preset_list()

    # ---------------- saving ----------------
    def mark_done(self):
        """Validate the classification of the current image, save and move to the next one."""
        if not self.annotations:
            return
        self.current_ann().done = True
        logger.info("Annotation %d validated: %s", self.index, self.current_ann().image_name)
        self.save()
        self.next_image()

    def save(self):
        """Write the status CSV (always at the same place, overwritten each time) and
        the detailed state that allows resuming the classification."""
        if not self.annotations:
            return
        try:
            csv_io.export_status_csv(self.annotations, self.config, self.csv_path)
            csv_io.save_state(self.annotations, self.state_path)
            self.set_status(f"Saved: {self.csv_path}")
        except Exception:
            logger.exception("Save failed")
            self.set_status("Save failed (see logs)")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Save before quitting?",
            "Do you want to save the classification before quitting?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            event.ignore()
            return
        if reply == QMessageBox.Yes:
            self.save()
        event.accept()
