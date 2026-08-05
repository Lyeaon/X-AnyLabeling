import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from anylabeling.views.labeling.utils.image import (
    band_to_array,
    band_to_pil,
    band_to_qimage,
)

try:
    import cv2
    from PyQt6 import QtWidgets

    import anylabeling.config as config_mod
    from anylabeling.config import get_config
    from anylabeling.views.mainwindow import MainWindow

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


def _make_band_image():
    """Create a 3-band BGR image with distinct per-band values."""
    h, w = 8, 8
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = 10  # band0 (depth)
    arr[:, :, 1] = 128  # band1 (reflectance)
    arr[:, :, 2] = 240  # band2 (other feature)
    return arr


class TestBandHelpers(unittest.TestCase):
    def test_band_to_array_extracts_correct_band(self):
        arr = _make_band_image()
        self.assertTrue(np.array_equal(band_to_array(arr, 0), np.full((8, 8), 10)))
        self.assertTrue(np.array_equal(band_to_array(arr, 1), np.full((8, 8), 128)))
        self.assertTrue(np.array_equal(band_to_array(arr, 2), np.full((8, 8), 240)))

    def test_band_to_array_normalizes_16bit(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint16)
        arr[:, :, 0] = np.array([0, 0, 10000, 65535] * 4).reshape(4, 4)
        out = band_to_array(arr, 0)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.min(), 0)
        self.assertEqual(out.max(), 255)

    def test_band_to_array_rejects_missing_band(self):
        self.assertIsNone(band_to_array(_make_band_image(), 5))
        self.assertIsNone(band_to_array(None, 0))

    def test_band_to_pil_returns_grayscale(self):
        pil_img = band_to_pil(_make_band_image(), 1)
        self.assertIsNotNone(pil_img)
        self.assertEqual(pil_img.mode, "L")
        self.assertEqual(pil_img.size, (8, 8))

    def test_band_to_qimage_returns_valid_qimage(self):
        qimg = band_to_qimage(_make_band_image(), 0)
        self.assertFalse(qimg.isNull())
        self.assertEqual(qimg.width(), 8)
        self.assertEqual(qimg.height(), 8)


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for channel view tests")
class TestChannelView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_mod.current_config_file = "{}"
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _build_widget(self, image_path):
        config = get_config("{}", {}, show_msg=False)
        window = MainWindow(self.app, config=config, filename=image_path)
        self.app.processEvents()
        return window.labeling_widget.view

    def test_channel_switching_and_side_by_side(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "source.png")
            cv2.imwrite(image_path, _make_band_image())

            widget = self._build_widget(image_path)
            try:
                self.assertTrue(widget._has_channels)
                self.assertIsNone(widget._main_channel)
                self.assertTrue(widget._channel_original_action.isChecked())

                widget.set_main_channel(0)
                self.app.processEvents()
                self.assertEqual(widget._main_channel, 0)
                self.assertTrue(widget._main_channel_actions[0].isChecked())
                self.assertFalse(widget._channel_original_action.isChecked())
                self.assertFalse(widget.canvas.pixmap.isNull())

                widget.set_main_channel(1)
                self.app.processEvents()
                self.assertEqual(widget._main_channel, 1)

                widget.set_main_channel(0)
                widget.toggle_side_by_side(True)
                self.app.processEvents()
                self.assertTrue(widget._side_by_side)
                self.assertIsNotNone(widget.canvas.compare_pixmap)
                self.assertEqual(widget.canvas.compare_pixmap.width(), 8)

                widget.set_compare_channel(2)
                self.app.processEvents()
                self.assertEqual(widget._compare_channel, 2)

                widget.close_compare_view()
                self.app.processEvents()
                self.assertFalse(widget._side_by_side)
                self.assertIsNone(widget.canvas.compare_pixmap)

                widget.set_main_channel(None)
                self.app.processEvents()
                self.assertIsNone(widget._main_channel)
                self.assertTrue(widget._channel_original_action.isChecked())

                widget.toggle_actions(True)
                self.assertTrue(widget._main_channel_actions[0].isEnabled())

                widget.reset_state()
                self.assertFalse(widget._has_channels)
                self.assertIsNone(widget._main_channel)
                self.assertIsNone(widget.channel_img)
            finally:
                widget.close()

    def test_normal_image_keeps_original_view(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "photo.png")
            cv2.imwrite(image_path, np.full((4, 4, 3), 120, dtype=np.uint8))

            widget = self._build_widget(image_path)
            try:
                self.assertTrue(widget._has_channels)
                self.assertIsNone(widget._main_channel)
                self.assertEqual(widget.canvas.pixmap.width(), 4)
                widget.toggle_actions(True)
                self.assertTrue(widget._main_channel_actions[0].isEnabled())
                widget.set_main_channel(0)
                self.app.processEvents()
                self.assertEqual(widget._main_channel, 0)
                self.assertEqual(widget.canvas.pixmap.width(), 4)
            finally:
                widget.close()


if __name__ == "__main__":
    unittest.main()
