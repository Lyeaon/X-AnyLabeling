import os
import unittest

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
class TestOrientationLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()
        from anylabeling.views.labeling.view3d.gl_widget import GLViewerWidget
        cls.widget = GLViewerWidget()

    def test_set_orientation_labels_adds_four_items(self):
        self.widget.clear_orientation()
        self.widget.set_orientation_labels(-1.5, 1.5, -1.5, 1.5, z_base=0.0)
        self.assertEqual(len(self.widget._orientation_items), 4)

    def test_labels_outside_edges(self):
        """Labels must sit just outside the footprint edge midpoints, offset by
        10% of the larger span, and at the requested height."""
        self.widget.set_orientation_labels(-3.0, 3.0, -2.0, 2.0, z_base=0.25)
        texts = {}
        for item in self.widget._orientation_items:
            pos = item.pos
            texts[item.text] = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3))
        pad = 0.10 * 6.0  # 10% of max span (6)
        self.assertEqual(texts["Image Left"], (-3.0 - pad, 0.0, 0.25))
        self.assertEqual(texts["Image Right"], (3.0 + pad, 0.0, 0.25))
        self.assertEqual(texts["Image Top"], (0.0, -2.0 - pad, 0.25))
        self.assertEqual(texts["Image Bottom"], (0.0, 2.0 + pad, 0.25))

    def test_labels_are_opaque_and_colored(self):
        """Regression: labels must use an opaque 0-255 int color. A float tuple
        made mkColor produce red=0, alpha=1 (invisible text)."""
        self.widget.set_orientation_labels(-1.5, 1.5, -1.5, 1.5, z_base=0.0)
        self.assertEqual(len(self.widget._orientation_items), 4)
        for item in self.widget._orientation_items:
            color = item.color
            self.assertEqual(color.alpha(), 255)
            self.assertGreater(color.red(), 0)

    def test_clear_orientation_empties_items(self):
        self.widget.set_orientation_labels(-1, 1, -1, 1, z_base=0)
        self.assertGreater(len(self.widget._orientation_items), 0)
        self.widget.clear_orientation()
        self.assertEqual(self.widget._orientation_items, [])

    def test_toggle_orientation_flips_visibility(self):
        self.widget.set_orientation_labels(-1, 1, -1, 1, z_base=0.0)
        self.widget.toggle_orientation(False)
        self.assertTrue(
            all(not item.visible() for item in self.widget._orientation_items)
        )
        self.widget.toggle_orientation(True)
        self.assertTrue(
            all(item.visible() for item in self.widget._orientation_items)
        )

    def test_clear_removes_orientation(self):
        self.widget.set_orientation_labels(-1, 1, -1, 1, z_base=0.0)
        self.widget.clear()
        self.assertEqual(self.widget._orientation_items, [])


if __name__ == "__main__":
    unittest.main()