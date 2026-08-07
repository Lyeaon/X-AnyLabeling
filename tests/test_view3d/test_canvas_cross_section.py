import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mouse_move(pos):
    return QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(pos),
        QtCore.QPointF(pos),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def _mouse_press(pos):
    return QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(pos),
        QtCore.QPointF(pos),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def _mouse_release(pos):
    return QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonRelease,
        QtCore.QPointF(pos),
        QtCore.QPointF(pos),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


class TestCanvasCrossSection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()
        from anylabeling.views.labeling.widgets.canvas import Canvas

        cls.canvas = Canvas(parent=None)
        pm = QtGui.QPixmap(100, 60)
        pm.fill(QtGui.QColor("gray"))
        cls.canvas.resize(100, 60)  # match pixmap => scale/offset map 1:1
        cls.canvas.load_pixmap(pm, clear_shapes=True)

    def test_crosshair_defaults_to_center(self):
        u, v = self.canvas._cross_section
        self.assertEqual((u, v), (50, 30))

    def test_crosshair_inactive_by_default(self):
        self.assertFalse(self.canvas._cross_section_active)

    def test_set_cross_section_active_locks_state(self):
        self.canvas.set_cross_section_active(True)
        self.assertTrue(self.canvas._cross_section_active)
        # Initializes to center if not already set.
        self.assertEqual(self.canvas._cross_section, (50, 30))
        self.canvas.set_cross_section_active(False)
        self.assertFalse(self.canvas._cross_section_active)
        self.assertFalse(self.canvas._cs_dragging)

    def test_drag_requires_active_mode(self):
        """Plain (inactive) labeling must NOT enter crosshair drag."""
        self.canvas.set_cross_section_active(False)
        self.canvas._cross_section = (50, 30)
        self.canvas.mousePressEvent(_mouse_press(QtCore.QPointF(50, 30)))
        self.assertFalse(self.canvas._cs_dragging)

    def test_drag_when_active(self):
        self.canvas.set_cross_section_active(True)
        self.canvas._cross_section = (50, 30)
        # Press within grab radius of the midpoint -> starts drag.
        self.canvas.mousePressEvent(_mouse_press(QtCore.QPointF(50, 30)))
        self.assertTrue(self.canvas._cs_dragging)
        self.canvas.mouseMoveEvent(_mouse_move(QtCore.QPointF(20, 40)))
        self.assertEqual(self.canvas._cross_section, (20, 40))
        self.canvas.mouseReleaseEvent(_mouse_release(QtCore.QPointF(20, 40)))
        self.assertFalse(self.canvas._cs_dragging)
        # Cross stays fixed afterwards (plain move does not change it).
        self.canvas.mouseMoveEvent(_mouse_move(QtCore.QPointF(90, 40)))
        self.assertEqual(self.canvas._cross_section, (20, 40))
        self.canvas.set_cross_section_active(False)

    def test_press_far_from_center_does_not_drag(self):
        self.canvas.set_cross_section_active(True)
        self.canvas._cross_section = (50, 30)
        self.canvas.mousePressEvent(_mouse_press(QtCore.QPointF(90, 50)))
        self.assertFalse(self.canvas._cs_dragging)
        self.canvas.set_cross_section_active(False)

    def test_set_cross_section_updates_and_emits(self):
        received = []
        self.canvas.cross_section_changed.connect(
            lambda u, v: received.append((u, v))
        )
        self.canvas.set_cross_section(12, 7)
        self.assertEqual(self.canvas._cross_section, (12, 7))
        self.assertEqual(received[-1], (12, 7))

    def test_set_cross_section_clamps(self):
        self.canvas.set_cross_section(5000, -3, emit=False)
        self.assertEqual(self.canvas._cross_section, (99, 0))


if __name__ == "__main__":
    unittest.main()