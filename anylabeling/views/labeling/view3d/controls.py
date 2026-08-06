"""3D Viewer controls and toolbar."""

from PyQt6 import QtCore, QtGui, QtWidgets


class View3DToolbar(QtWidgets.QToolBar):
    """Toolbar for 3D viewer controls."""

    # Signals
    reset_view_requested = QtCore.pyqtSignal()
    fit_view_requested = QtCore.pyqtSignal()
    export_requested = QtCore.pyqtSignal(str)
    point_size_changed = QtCore.pyqtSignal(float)
    point_size_mode_changed = QtCore.pyqtSignal(bool)  # True = px mode, False = relative
    background_color_changed = QtCore.pyqtSignal(str)  # hex color
    grid_toggled = QtCore.pyqtSignal(bool)
    axis_toggled = QtCore.pyqtSignal(bool)
    projection_changed = QtCore.pyqtSignal(bool)  # True = perspective, False = ortho
    color_mode_changed = QtCore.pyqtSignal(str)  # depth, reflectance, other, uniform
    point_size_value_changed = QtCore.pyqtSignal(int)
    depth_scale_changed = QtCore.pyqtSignal(float)  # depth scale factor

    def __init__(self, parent=None):
        super().__init__("3D View Controls", parent)
        self.setObjectName("View3DToolbar")
        self.setMovable(False)
        self.setFloatable(False)
        self._init_ui()

    def _init_ui(self):
        # View controls group
        self._add_action("Reset View", "view-refresh", "Reset camera to default position",
                         self.reset_view_requested.emit)
        self._add_action("Fit to Data", "view-fit", "Fit camera to show all data",
                         self.fit_view_requested.emit)

        self.addSeparator()

        # Projection mode
        self.proj_action = self._add_checkable_action(
            "Perspective", "view-perspective", "Toggle perspective/orthographic projection",
            True, self._on_projection_changed
        )

        self.addSeparator()

        # Point size control
        self.size_label = QtWidgets.QLabel("Point Size:")
        self.size_label.setMinimumWidth(80)
        self.addWidget(self.size_label)

        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(1, 20)
        self.size_spin.setValue(3)
        self.size_spin.setFixedWidth(60)
        self.size_spin.valueChanged.connect(self.point_size_value_changed.emit)
        self.addWidget(self.size_spin)

        self.px_mode_action = self._add_checkable_action(
            "Pixel Size", None, "Point size in pixels (vs relative)",
            True, self.point_size_mode_changed.emit
        )

        self.addSeparator()

        # Color mode
        self.color_mode_combo = QtWidgets.QComboBox()
        self.color_mode_combo.addItems([
            "Depth (colormap)",
            "Reflectance",
            "Other Feature",
            "Uniform",
            "RGB (if available)"
        ])
        self.color_mode_combo.setToolTip("Colorization mode for point cloud")
        self.color_mode_combo.currentTextChanged.connect(self.color_mode_changed.emit)
        self.addWidget(QtWidgets.QLabel("Color:"))
        self.addWidget(self.color_mode_combo)

        self.addSeparator()

        # Depth scale control
        self.depth_scale_label = QtWidgets.QLabel("Depth Scale:")
        self.depth_scale_label.setMinimumWidth(80)
        self.addWidget(self.depth_scale_label)

        self.depth_scale_combo = QtWidgets.QComboBox()
        self.depth_scale_combo.addItems([
            "Auto-detect",
            "0-255 = 1m (255)",
            "0-255 = 10m (25.5)",
            "0-255 = 100m (2.55)",
            "0-255 = 1000m (0.255)",
            "Custom..."
        ])
        self.depth_scale_combo.setToolTip("Depth scale: how to interpret 0-255 depth values")
        self.depth_scale_combo.currentTextChanged.connect(self._on_depth_scale_changed)
        self.addWidget(self.depth_scale_combo)

        self.addSeparator()

        # Visual toggles
        self.grid_action = self._add_checkable_action(
            "Grid", "view-grid", "Show/hide grid", True,
            self.grid_toggled.emit
        )
        self.axis_action = self._add_checkable_action(
            "Axis", "view-axis", "Show/hide coordinate axis", True,
            self.axis_toggled.emit
        )

        self.addSeparator()

        # Background color
        self.bg_color_btn = QtWidgets.QPushButton("BG Color")
        self.bg_color_btn.setFixedWidth(80)
        self.bg_color_btn.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.bg_color_btn.setToolTip("Change background color")
        self.bg_color_btn.clicked.connect(self._choose_bg_color)
        self.addWidget(self.bg_color_btn)

        self.addSeparator()

        # Export
        self._add_action("Export PLY", "document-save", "Export point cloud as PLY",
                         lambda: self.export_requested.emit("ply"))
        self._add_action("Export OBJ", "document-save", "Export mesh as OBJ",
                         lambda: self.export_requested.emit("obj"))
        self._add_action("Export NPZ", "document-save", "Export as NumPy arrays",
                         lambda: self.export_requested.emit("npz"))

        self.addSeparator()

        # Close button
        self.close_action = self._add_action(
            "Close", "window-close", "Close 3D viewer",
            lambda: self.parent().close() if self.parent() else None
        )

    def _add_action(self, text: str, icon_name: str, tooltip: str, callback) -> QtGui.QAction:
        """Add a simple action button."""
        action = QtGui.QAction(text, self)
        action.setToolTip(tooltip)
        action.triggered.connect(callback)
        self.addAction(action)
        return action

    def _add_checkable_action(self, text: str, icon_name: str, tooltip: str,
                              checked: bool, callback) -> QtGui.QAction:
        """Add a checkable action button."""
        action = QtGui.QAction(text, self)
        action.setToolTip(tooltip)
        action.setCheckable(True)
        action.setChecked(checked)
        action.toggled.connect(callback)
        self.addAction(action)
        return action

    def _on_projection_changed(self, checked: bool):
        """Handle projection mode change."""
        self.projection_changed.emit(checked)

    def _on_depth_scale_changed(self, text: str):
        """Handle depth scale change from combo box."""
        scale_map = {
            "Auto-detect": 0.0,  # Special value for auto-detect
            "0-255 = 1m (255)": 255.0,
            "0-255 = 10m (25.5)": 25.5,
            "0-255 = 100m (2.55)": 2.55,
            "0-255 = 1000m (0.255)": 0.255,
        }
        if text in scale_map:
            self.depth_scale_changed.emit(scale_map[text])
        elif text == "Custom...":
            # TODO: Show custom input dialog
            pass

    def _choose_bg_color(self):
        """Open color picker for background color."""
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor

        current = self.bg_color_btn.palette().button().color()
        color = QColorDialog.getColor(current, self, "Background Color")
        if color.isValid():
            hex_color = color.name()
            self.bg_color_btn.setStyleSheet(f"background-color: {hex_color}; color: white;")
            self.background_color_changed.emit(hex_color)

    def set_projection_mode(self, perspective: bool):
        """Set projection mode (programmatically)."""
        self.proj_action.setChecked(perspective)

    def set_grid_visible(self, visible: bool):
        self.grid_action.setChecked(visible)

    def set_axis_visible(self, visible: bool):
        self.axis_action.setChecked(visible)

    def set_color_mode(self, mode: str):
        """Set color mode by string."""
        index = self.color_mode_combo.findText(mode)
        if index >= 0:
            self.color_mode_combo.setCurrentIndex(index)

    def set_point_size(self, size: int):
        self.size_spin.setValue(size)