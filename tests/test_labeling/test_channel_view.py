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
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

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
                self.assertFalse(widget._channel_scroll_area.isHidden())
                self.assertFalse(widget.channel_canvas.pixmap.isNull())
                self.assertEqual(widget.channel_canvas.pixmap.width(), 8)

                widget.set_compare_channel(2)
                self.app.processEvents()
                self.assertEqual(widget._compare_channel, 2)

                # Both canvases share the same authoritative shapes model.
                self.assertIs(widget.channel_canvas.shapes, widget.canvas.shapes)

                widget.close_compare_view()
                self.app.processEvents()
                self.assertFalse(widget._side_by_side)
                self.assertTrue(widget._channel_scroll_area.isHidden())

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

    def test_shared_shapes_and_live_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "source.png")
            cv2.imwrite(image_path, _make_band_image())

            widget = self._build_widget(image_path)
            try:
                widget._main_channel = 0
                widget.toggle_side_by_side(True)
                self.app.processEvents()
                self.assertIs(widget.channel_canvas.shapes, widget.canvas.shapes)
                self.assertTrue(widget.channel_live_timer.isActive())
                # Both canvases share the same authoritative shapes model.
                from anylabeling.views.labeling.shape import Shape
                from PyQt6.QtCore import QPointF

                shape = Shape()
                shape.add_point(QPointF(1, 1))
                shape.add_point(QPointF(3, 1))
                shape.add_point(QPointF(3, 3))
                shape.add_point(QPointF(1, 3))
                shape.shape_type = "rectangle"
                shape.closed = True
                widget.canvas.shapes.append(shape)
                widget._repoint_shapes()
                self.app.processEvents()
                self.assertIs(widget.channel_canvas.shapes, widget.canvas.shapes)
                self.assertEqual(len(widget.channel_canvas.shapes), 1)
                widget.toggle_side_by_side(False)
                self.app.processEvents()
                self.assertFalse(widget.channel_live_timer.isActive())
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

    def _enable_side_by_side(self, widget):
        widget._main_channel = 0
        widget.toggle_side_by_side(True)
        self.app.processEvents()
        self.assertIs(widget.channel_canvas.shapes, widget.canvas.shapes)
        self.assertTrue(widget.channel_live_timer.isActive())

    def _append_shape(self, widget):
        from anylabeling.views.labeling.shape import Shape

        shape = Shape()
        shape.add_point(QPointF(1, 1))
        shape.add_point(QPointF(3, 1))
        shape.add_point(QPointF(3, 3))
        shape.add_point(QPointF(1, 3))
        shape.shape_type = "rectangle"
        shape.closed = True
        widget.canvas.shapes.append(shape)
        widget._repoint_shapes()
        return shape

    def test_event_filter_tracks_input_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "source.png")
            cv2.imwrite(image_path, _make_band_image())

            widget = self._build_widget(image_path)
            try:
                self.assertIsNone(widget._last_input_canvas)
                press = QMouseEvent(
                    QEvent.Type.MouseButtonPress,
                    QPointF(1, 1),
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                widget.eventFilter(widget.channel_canvas, press)
                self.assertIs(widget._last_input_canvas, widget.channel_canvas)
                move = QMouseEvent(
                    QEvent.Type.MouseMove,
                    QPointF(2, 2),
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                widget.eventFilter(widget.canvas, move)
                self.assertIs(widget._last_input_canvas, widget.canvas)
                widget.reset_state()
                self.assertIsNone(widget._last_input_canvas)
            finally:
                widget.close()

    def test_live_sync_direction_stays_on_input_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "source.png")
            cv2.imwrite(image_path, _make_band_image())

            widget = self._build_widget(image_path)
            try:
                self._enable_side_by_side(widget)
                shape = self._append_shape(widget)
                main = widget.canvas
                ch = widget.channel_canvas

                # User drags a shape on the RIGHT (channel) canvas.
                ch.moving_shape = True
                ch.h_shape = shape
                ch.offsets = (QPointF(2, 2), QPointF(2, 2))
                widget._last_input_canvas = ch

                widget._channel_sync_tick()
                # Transient state is mirrored onto main for live rendering...
                self.assertTrue(main.moving_shape)
                self.assertIs(main.h_shape, shape)
                # ...but the sync direction must NOT flip to main->channel
                # just because main now carries the mirrored flags.
                self.assertIs(widget._active_editing_canvas(), ch)

                widget._channel_sync_tick()
                self.assertIs(widget._active_editing_canvas(), ch)
                self.assertIs(main.h_shape, shape)

                # Gesture ends (mouse release clears the source flags).
                ch.moving_shape = False
                ch.h_shape = None
                ch.offsets = None
                ch.current = None
                ch.is_move_editing = False
                widget._channel_sync_tick()
                self.assertIsNone(widget._active_editing_canvas())
                # Stale mirrored flags on main are cleaned up (no ghost).
                self.assertFalse(main.moving_shape)
                self.assertIsNone(main.h_shape)
                # offsets is restored to the default 2-tuple, never None, so
                # canvas move code (reads ``offsets[0]`` unguarded) can't crash.
                self.assertIsInstance(main.offsets, tuple)
                self.assertEqual(len(main.offsets), 2)

                # Same guarantees in the opposite direction (main drag).
                main.moving_shape = True
                main.h_shape = shape
                main.offsets = (QPointF(3, 3), QPointF(3, 3))
                widget._last_input_canvas = main
                widget._channel_sync_tick()
                self.assertTrue(ch.moving_shape)
                self.assertIs(widget._active_editing_canvas(), main)
                widget._channel_sync_tick()
                self.assertIs(widget._active_editing_canvas(), main)

                main.moving_shape = False
                main.h_shape = None
                main.offsets = None
                widget._channel_sync_tick()
                self.assertIsNone(widget._active_editing_canvas())
                self.assertFalse(ch.moving_shape)
                self.assertIsNone(ch.h_shape)
            finally:
                widget.close()

    def test_navigation_sync_zoom_and_scroll(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "large.png")
            cv2.imwrite(
                image_path, np.zeros((400, 500, 3), dtype=np.uint8)
            )

            widget = self._build_widget(image_path)
            try:
                self._enable_side_by_side(widget)
                widget.show()
                self.app.processEvents()

                # Zoom must reach both canvases.
                widget.set_zoom(200)
                self.app.processEvents()
                self.assertEqual(
                    widget.channel_canvas.scale, widget.canvas.scale
                )

                # Main scroll position must reach the channel scroll area once
                # the deferred layout pass has run.
                hbar = widget.scroll_bars[Qt.Orientation.Horizontal]
                chbar = widget._channel_scroll_area.horizontalScrollBar()
                if hbar.maximum() > 0:
                    target = min(7, hbar.maximum())
                    hbar.setValue(target)
                    self.app.processEvents()
                    self.assertEqual(chbar.value(), target)
                    # The channel pane must actually scroll (viewport moves),
                    # not merely track the scrollbar value.
                    self.assertLess(
                        widget._channel_scroll_area.widget().pos().x(), 0
                    )
                    vbar = widget.scroll_bars[Qt.Orientation.Vertical]
                    cvbar = widget._channel_scroll_area.verticalScrollBar()
                    if vbar.maximum() > 0:
                        vtarget = min(5, vbar.maximum())
                        vbar.setValue(vtarget)
                        self.app.processEvents()
                        self.assertEqual(cvbar.value(), vtarget)
            finally:
                widget.close()

    def test_offsets_never_set_to_none(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "source.png")
            cv2.imwrite(image_path, _make_band_image())

            widget = self._build_widget(image_path)
            try:
                self._enable_side_by_side(widget)
                shape = self._append_shape(widget)
                main = widget.canvas
                ch = widget.channel_canvas

                # A channel drag mirrors transient state onto main...
                ch.moving_shape = True
                ch.h_shape = shape
                ch.offsets = (QPointF(2, 2), QPointF(2, 2))
                widget._last_input_canvas = ch
                widget._channel_sync_tick()
                self.assertIsInstance(main.offsets, tuple)
                self.assertEqual(len(main.offsets), 2)

                # ...then the gesture ends and stale mirrors are cleared. The
                # receiving canvas must get the default 2-tuple back, not None.
                ch.moving_shape = False
                ch.h_shape = None
                ch.offsets = None
                ch.current = None
                ch.is_move_editing = False
                widget._channel_sync_tick()
                self.assertIsNone(widget._active_editing_canvas())
                self.assertIsInstance(main.offsets, tuple)
                self.assertEqual(len(main.offsets), 2)
                self.assertFalse(main.moving_shape)

                # A subsequent move on main must not raise even though the
                # canvas was just cleared by the sync machinery.
                widget.canvas.bounded_move_shapes([shape], QPointF(1, 1))
            finally:
                widget.close()

    def test_mode_sync_between_canvases(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "source.png")
            cv2.imwrite(image_path, _make_band_image())

            widget = self._build_widget(image_path)
            try:
                self._enable_side_by_side(widget)
                from anylabeling.views.labeling.widgets.canvas import Canvas

                # Enter rectangle-create mode on the left -> right must follow.
                widget.toggle_draw_mode(False, create_mode="rectangle")
                self.assertEqual(widget.canvas.mode, Canvas.CREATE)
                self.assertEqual(widget.channel_canvas.mode, Canvas.CREATE)
                self.assertEqual(widget.channel_canvas.create_mode, "rectangle")

                # Back to edit mode -> right must follow.
                widget.toggle_draw_mode(True)
                self.assertEqual(widget.canvas.mode, Canvas.EDIT)
                self.assertEqual(widget.channel_canvas.mode, Canvas.EDIT)

                # Another create mode -> right must follow.
                widget.toggle_draw_mode(False, create_mode="polygon")
                self.assertEqual(widget.channel_canvas.create_mode, "polygon")
                self.assertEqual(widget.channel_canvas.mode, Canvas.CREATE)
            finally:
                widget.close()

    def test_selection_sync_between_canvases(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "source.png")
            cv2.imwrite(image_path, _make_band_image())

            widget = self._build_widget(image_path)
            try:
                self._enable_side_by_side(widget)
                from anylabeling.views.labeling.shape import Shape

                a = Shape()
                for p in [(10, 10), (12, 10), (12, 12), (10, 12)]:
                    a.add_point(QPointF(*p))
                a.shape_type = "rectangle"
                a.closed = True
                b = Shape()
                for p in [(20, 20), (22, 20), (22, 22), (20, 22)]:
                    b.add_point(QPointF(*p))
                b.shape_type = "rectangle"
                b.closed = True
                widget.canvas.shapes.extend([a, b])
                widget._repoint_shapes()

                # Selecting through the label list updates both canvases.
                widget.shape_selection_changed([b])
                self.assertIs(
                    widget.channel_canvas.selected_shapes,
                    widget.canvas.selected_shapes,
                )
                self.assertEqual(widget.channel_canvas.selected_shapes, [b])

                # Selecting a shape on the RIGHT canvas updates both + the
                # label list path.
                widget._channel_selection_changed([a])
                self.assertIs(
                    widget.channel_canvas.selected_shapes,
                    widget.canvas.selected_shapes,
                )
                self.assertEqual(widget.channel_canvas.selected_shapes, [a])

                # A drag on the right moves the canvas's own selection (a),
                # not the previously selected shape (b).
                widget.channel_canvas.prev_point = QPointF(0, 0)
                widget.channel_canvas.bounded_move_shapes(
                    widget.channel_canvas.selected_shapes, QPointF(5, 5)
                )
                self.assertEqual(a[0], QPointF(15, 15))
                self.assertEqual(b[0], QPointF(20, 20))
            finally:
                widget.close()


if __name__ == "__main__":
    unittest.main()
