import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


try:
    import pyqtgraph.opengl as gl
    PYQTGRAPH_AVAILABLE = gl is not None
except ImportError:
    PYQTGRAPH_AVAILABLE = False


@unittest.skipUnless(PYQTGRAPH_AVAILABLE, "pyqtgraph is required for GL tests")
class TestCrossSlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()
        from anylabeling.views.labeling.view3d.gl_widget import GLViewerWidget
        cls.widget = GLViewerWidget()

    def test_set_cross_slice_adds_two_items(self):
        self.widget.set_cross_slice(0.0, 0.0, -1.5, 1.5, -1.5, 1.5, z=0.5)
        self.assertEqual(len(self.widget._cross_items), 2)

    def test_lines_have_constant_coordinates(self):
        """The X=x0 line stays at constant X; the Y=y0 line stays at Y."""
        self.widget.set_cross_slice(0.25, -0.5, -1.5, 1.5, -1.5, 1.5, z=0.5)
        items = self.widget._cross_items
        self.assertEqual(len(items), 2)
        # One item spans the X axis (varied X, constant Y), the other spans Y.
        x_span, y_span = 0, 0
        for item in items:
            pts = item.pos
            xs = pts[:, 0]
            ys = pts[:, 1]
            if np.ptp(xs) > np.ptp(ys):
                x_span += 1
            else:
                y_span += 1
        self.assertEqual((x_span, y_span), (1, 1))

    def test_clear_cross_slice_empties(self):
        self.widget.set_cross_slice(0.0, 0.0, -1, 1, -1, 1, z=0.0)
        self.assertGreater(len(self.widget._cross_items), 0)
        self.widget.clear_cross_slice()
        self.assertEqual(self.widget._cross_items, [])

    def test_clear_removes_cross(self):
        self.widget.set_cross_slice(0.0, 0.0, -1, 1, -1, 1, z=0.0)
        self.widget.clear()
        self.assertEqual(self.widget._cross_items, [])

    def test_dashed_points_segments_even(self):
        pts = self.widget._dashed_points((0, 0, 0), (1, 0, 0))
        self.assertEqual(len(pts) % 2, 0)
        self.assertGreater(len(pts), 0)


if __name__ == "__main__":
    unittest.main()