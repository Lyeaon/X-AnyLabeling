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
    orientation_toggled = QtCore.pyqtSignal(bool)
    projection_changed = QtCore.pyqtSignal(bool)  # True = perspective, False = ortho
    color_mode_changed = QtCore.pyqtSignal(str)  # depth, reflectance, other, uniform
    point_size_value_changed = QtCore.pyqtSignal(int)
    depth_scale_changed = QtCore.pyqtSignal(float)  # depth scale factor
    fov_changed = QtCore.pyqtSignal(float)  # field of view in degrees
    depth_is_range_changed = QtCore.pyqtSignal(bool)  # True = Euclidean range, False = perpendicular Z
    reconstruction_mode_changed = QtCore.pyqtSignal(str)  # "pinhole" | "orthographic"
    field_size_changed = QtCore.pyqtSignal(float, float)  # (width_m, height_m)
    render_requested = QtCore.pyqtSignal()  # apply parameter changes and rebuild
    cross_section_changed = QtCore.pyqtSignal(float, float)  # (col u, row v)

    def __init__(self, parent=None):
        super().__init__("3D View Controls", parent)
        self.setObjectName("View3DToolbar")
        self.setMovable(False)
        self.setFloatable(False)
        self._custom_depth_prompt_active = False
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

        # Field of view control
        self.fov_label = QtWidgets.QLabel("FOV:")
        self.fov_label.setMinimumWidth(60)
        self.addWidget(self.fov_label)

        self.fov_spin = QtWidgets.QDoubleSpinBox()
        self.fov_spin.setRange(5.0, 180.0)
        self.fov_spin.setDecimals(1)
        self.fov_spin.setSingleStep(1.0)
        self.fov_spin.setValue(60.0)
        self.fov_spin.setSuffix("\u00b0")
        self.fov_spin.setToolTip(
            "Field of view used to reconstruct depth. For a flat surface, "
            "adjust until the point cloud is planar (plane-fit R² near 1)."
        )
        self.fov_spin.valueChanged.connect(self.fov_changed.emit)
        self.addWidget(self.fov_spin)

        self.addSeparator()

        # Depth convention: Euclidean range vs perpendicular Z
        self.range_action = self._add_checkable_action(
            "Depth is Range", None,
            "Treat depth values as Euclidean range (radial distance along each "
            "pixel's ray) and convert to perpendicular Z. Enables flat surfaces "
            "to reconstruct as flat planes instead of a pyramid.",
            True, self.depth_is_range_changed.emit
        )

        self.addSeparator()

        # Reconstruction projection: pinhole vs orthographic (top-view)
        self.recon_proj_combo = QtWidgets.QComboBox()
        self.recon_proj_combo.addItems([
            "Orthographic (top-view)",
            "Pinhole (forward)"
        ])
        self.recon_proj_combo.setCurrentIndex(0)
        self.recon_proj_combo.setToolTip(
            "How the depth channel is turned into 3D points. Use Orthographic "
            "(top-view) for nadir/overhead sensors where each pixel maps directly "
            "to a ground cell and the value is height above ground - this keeps "
            "a flat road flat instead of folding it into a pyramid."
        )
        self.recon_proj_combo.currentIndexChanged.connect(self._on_recon_proj_changed)
        self.addWidget(QtWidgets.QLabel("Projection:"))
        self.addWidget(self.recon_proj_combo)

        # Physical footprint (width/height) for orthographic
        self.field_w_label = QtWidgets.QLabel("Width (m):")
        self.field_w_label.setMinimumWidth(60)
        self.addWidget(self.field_w_label)

        self.field_w_spin = QtWidgets.QDoubleSpinBox()
        self.field_w_spin.setRange(0.001, 1000.0)
        self.field_w_spin.setDecimals(4)
        self.field_w_spin.setValue(3.0)
        self.field_w_spin.setToolTip(
            "Physical width of the orthographic footprint in meters (X axis). "
            "Set to the real size of the captured area (e.g. 3 m)."
        )
        self.field_w_spin.valueChanged.connect(self._on_field_size_spin_changed)
        self.addWidget(self.field_w_spin)

        self.field_h_label = QtWidgets.QLabel("Height (m):")
        self.field_h_label.setMinimumWidth(68)
        self.addWidget(self.field_h_label)

        self.field_h_spin = QtWidgets.QDoubleSpinBox()
        self.field_h_spin.setRange(0.001, 1000.0)
        self.field_h_spin.setDecimals(4)
        self.field_h_spin.setValue(3.0)
        self.field_h_spin.setToolTip(
            "Physical height of the orthographic footprint in meters (Y axis). "
            "Set to the real size of the observed area (e.g. 3 m)."
        )
        self.field_h_spin.valueChanged.connect(self._on_field_size_spin_changed)
        self.addWidget(self.field_h_spin)

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
        self.orientation_action = self._add_checkable_action(
            "Orientation", None,
            "Show image-direction labels (Left/Right/Top/Bottom)", True,
            self.orientation_toggled.emit
        )

        self.addSeparator()

        # Cross-section crosshair position (col u, row v)
        self.cross_label = QtWidgets.QLabel("Cross:")
        self.addWidget(self.cross_label)
        self.cs_u_spin = QtWidgets.QSpinBox()
        self.cs_u_spin.setRange(0, 1_000_000)
        self.cs_u_spin.setValue(0)
        self.cs_u_spin.setToolTip("Crosshair column (image x / u)")
        self.cs_u_spin.valueChanged.connect(self._on_cross_section_spin_changed)
        self.addWidget(self.cs_u_spin)
        self.cs_v_spin = QtWidgets.QSpinBox()
        self.cs_v_spin.setRange(0, 1_000_000)
        self.cs_v_spin.setValue(0)
        self.cs_v_spin.setToolTip("Crosshair row (image y / v)")
        self.cs_v_spin.valueChanged.connect(self._on_cross_section_spin_changed)
        self.addWidget(self.cs_v_spin)

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

        # Render (apply parameter changes and rebuild point cloud)
        self._add_action(
            "Render", "view-refresh",
            "Apply the current parameter settings and rebuild the point cloud.",
            self.render_requested.emit
        )

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

    def _on_recon_proj_changed(self, index: int):
        """Handle reconstruction projection mode change."""
        if index == 1:
            self.reconstruction_mode_changed.emit("pinhole")
        else:
            self.reconstruction_mode_changed.emit("orthographic")

    def _on_field_size_spin_changed(self, _value: float):
        """Handle field width/height change - emit both current values."""
        self.field_size_changed.emit(
            self.field_w_spin.value(), self.field_h_spin.value()
        )

    def _on_cross_section_spin_changed(self, _value: int):
        """Handle crosshair position change from the spin-boxes."""
        self.cross_section_changed.emit(
            self.cs_u_spin.value(), self.cs_v_spin.value()
        )

    def set_cross_section(self, u: float, v: float):
        """Set the crosshair position programmatically (image pixels)."""
        self.cs_u_spin.blockSignals(True)
        self.cs_v_spin.blockSignals(True)
        self.cs_u_spin.setValue(int(u))
        self.cs_v_spin.setValue(int(v))
        self.cs_u_spin.blockSignals(False)
        self.cs_v_spin.blockSignals(False)

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
        elif text.startswith("Custom"):
            self._set_custom_depth_scale()

    def _set_custom_depth_scale(self):
        """Prompt the user for a max real-world depth (meters) and emit the
        corresponding depth scale (`255 / D`)."""
        # Guard against re-entry from the setCurrentIndex below re-triggering
        # currentTextChanged.
        if self._custom_depth_prompt_active:
            return
        self._custom_depth_prompt_active = True
        try:
            value, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Custom Depth",
                "Max real-world depth (meters).\n"
                "For a road scene, e.g. 0.5 means 0-255 maps to 0-0.5 m:",
                0.5,  # Initial value
                0.001,  # Min
                1000.0,  # Max
                3,  # Decimals
            )
            if not ok:
                return

            from .pointcloud import max_depth_meters_to_scale

            depth_scale = max_depth_meters_to_scale(value)
            # Reflect the chosen depth in the combo (rename the Custom item) so
            # the value is visible and preserved when switching presets.
            custom_text = f"Custom (D={value:g} m)"
            idx = self.depth_scale_combo.findText(custom_text)
            if idx < 0:
                # Reuse the existing Custom item (plain "Custom..." or a
                # previous custom value) rather than accumulating new entries.
                idx = self._find_custom_item_index()
                self.depth_scale_combo.setItemText(idx, custom_text)
            self.depth_scale_combo.setCurrentIndex(idx)
            self.depth_scale_changed.emit(depth_scale)
        finally:
            self._custom_depth_prompt_active = False

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