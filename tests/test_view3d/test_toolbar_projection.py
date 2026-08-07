import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestReconstructionProjectionToolbar(unittest.TestCase):
    """Verify the top-view/pinhole reconstruction mode control wiring."""

    @classmethod
    def setUpClass(cls):
        cls.app = _app()
        from anylabeling.views.labeling.view3d.controls import View3DToolbar
        cls.view3d_controls = View3DToolbar
        cls.toolbar = View3DToolbar()

    def test_default_is_orthographic(self):
        self.assertEqual(self.toolbar.recon_proj_combo.currentIndex(), 0)
        self.assertEqual(self.toolbar.recon_proj_combo.currentText(), "Orthographic (top-view)")
        self.assertEqual(self.toolbar.field_w_spin.value(), 3.0)
        self.assertEqual(self.toolbar.field_h_spin.value(), 3.0)

    def test_mode_signal_emits_pinhole_and_orthographic(self):
        emitted = []
        self.toolbar.reconstruction_mode_changed.connect(emitted.append)
        self.toolbar.recon_proj_combo.setCurrentIndex(1)
        self.toolbar.recon_proj_combo.setCurrentIndex(0)
        self.assertEqual(emitted, ["pinhole", "orthographic"])

    def test_field_size_signal_emits_pair(self):
        emitted = []
        self.toolbar.field_size_changed.connect(
            lambda w, h: emitted.append((w, h))
        )
        self.toolbar.field_w_spin.setValue(4.0)
        self.assertTrue(emitted)
        # last emission carries the current full pair (w, h)
        self.assertEqual(emitted[-1], (4.0, 3.0))
        self.toolbar.field_h_spin.setValue(2.5)
        self.assertEqual(emitted[-1], (4.0, 2.5))

    def test_param_change_does_not_request_render(self):
        """Changing reconstruction parameters must only update state, not
        trigger a rebuild. Only the Render action requests a render."""
        renders = []
        self.toolbar.render_requested.connect(lambda: renders.append(True))
        self.toolbar.recon_proj_combo.setCurrentIndex(0)  # already 0, no change
        self.toolbar.reconstruction_mode_changed.emit("pinhole")
        self.toolbar.field_size_changed.emit(50.0, 50.0)
        self.assertEqual(renders, [])

    def test_render_action_emits_render_requested(self):
        renders = []
        self.toolbar.render_requested.connect(lambda: renders.append(True))
        target = next(
            a for a in self.toolbar.actions()
            if a.text() == "Render"
        )
        target.trigger()
        self.assertEqual(renders, [True])


if __name__ == "__main__":
    unittest.main()