"""Main 3D viewer window for X-AnyLabeling."""

import os
import os.path as osp
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .gl_widget import GLViewerWidget
from .controls import View3DToolbar
from .pointcloud import (
    depth_to_pointcloud,
    pointcloud_to_arrays,
    create_pointcloud_from_depth_image,
    estimate_camera_params,
    planarity_r2,
    extract_cross_sections,
    OPEN3D_AVAILABLE,
)

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    pg = None


class View3DWindow(QtWidgets.QMainWindow):
    """Main 3D viewer window for depth/reflectance channel visualization."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        depth_channel: np.ndarray,
        reflectance_channel: Optional[np.ndarray] = None,
        other_channel: Optional[np.ndarray] = None,
        use_channel: str = "depth",
        focal_length: Optional[float] = None,
        depth_scale: float = 255.0,
        parent_widget=None,
    ):
        """Initialize 3D viewer window.

        Args:
            parent: Parent widget
            depth_channel: HxW depth map (normalized 0-1 or metric)
            reflectance_channel: Optional HxW reflectance map
            other_channel: Optional HxW other feature channel
            use_channel: Which channel to use for 3D ("depth", "reflectance", "other")
            focal_length: Optional focal length in pixels
            depth_scale: Scale factor to convert depth values to meters.
                         Default 255.0 for 8-bit (0-255 -> 0-1 meters).
                         Use 1000.0 for mm, 100.0 for cm, 1.0 for meters.
            parent_widget: Reference to parent LabelingWidget for settings
        """
        super().__init__(parent)
        self.setWindowTitle("3D Viewer")
        self.setMinimumSize(800, 600)
        self.resize(1024, 768)

        self._parent_widget = parent_widget
        self._depth_channel = depth_channel
        self._reflectance_channel = reflectance_channel
        self._other_channel = other_channel
        self._use_channel = use_channel
        self._focal_length = focal_length
        self._depth_scale = depth_scale
        self._fov_degrees = 60.0
        self._depth_is_range = True
        self._projection = "orthographic"
        self._field_width_m = 3.0
        self._field_height_m = 3.0
        self._show_orientation = True
        self._cross_u = 0
        self._cross_v = 0
        self._cross_active = False
        self._cross_zmin = 0.0
        self._cross_zmax = 0.0

        self._current_points = None
        self._current_colors = None
        self._current_pcd = None

        self._init_ui()
        self._generate_3d_model()
        if self._depth_channel is not None:
            h, w = self._depth_channel.shape[:2]
            self._cross_u, self._cross_v = w // 2, h // 2
            self._cross_active = True
            self._refresh_cross_section()

    def _init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("3D Viewer")
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.WindowMinMaxButtonsHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
        )

        # Central widget with layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar
        self.toolbar = View3DToolbar(self)
        self.toolbar.reset_view_requested.connect(self._on_reset_view)
        self.toolbar.fit_view_requested.connect(self._on_fit_view)
        self.toolbar.export_requested.connect(self._on_export)
        self.toolbar.point_size_changed.connect(self._on_point_size_changed)
        self.toolbar.point_size_mode_changed.connect(self._on_point_size_mode_changed)
        self.toolbar.background_color_changed.connect(self._on_bg_color_changed)
        self.toolbar.grid_toggled.connect(self._on_grid_toggled)
        self.toolbar.axis_toggled.connect(self._on_axis_toggled)
        self.toolbar.orientation_toggled.connect(self._on_orientation_toggled)
        self.toolbar.projection_changed.connect(self._on_projection_changed)
        self.toolbar.color_mode_changed.connect(self._on_color_mode_changed)
        self.toolbar.point_size_value_changed.connect(self._on_point_size_value_changed)
        self.toolbar.depth_scale_changed.connect(self._on_depth_scale_changed)
        self.toolbar.fov_changed.connect(self._on_fov_changed)
        self.toolbar.depth_is_range_changed.connect(self._on_depth_is_range_changed)
        self.toolbar.reconstruction_mode_changed.connect(self._on_reconstruction_mode_changed)
        self.toolbar.field_size_changed.connect(self._on_field_size_changed)
        self.toolbar.render_requested.connect(self._on_render_requested)
        self.toolbar.cross_section_changed.connect(self._on_toolbar_cross_section_changed)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # 3D Viewer (left) + cross-section profile panels (right)
        self.gl_widget = GLViewerWidget(self)
        self._build_profile_panels()
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._splitter.addWidget(self.gl_widget)
        self._splitter.addWidget(self._profile_container)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([720, 300])
        self.setCentralWidget(self._splitter)

        # Status bar
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Connect GL viewer signals
        self.gl_widget.mouse_moved.connect(self._on_mouse_moved)

        # Shortcuts
        self._setup_shortcuts()

    def _build_profile_panels(self):
        """Build the right-hand 1-D cross-section profile panels."""
        self._profile_container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(self._profile_container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("Cross-Section Profiles")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        if not PYQTGRAPH_AVAILABLE:
            placeholder = QtWidgets.QLabel("pyqtgraph required for profiles")
            placeholder.setWordWrap(True)
            layout.addWidget(placeholder)
            layout.addStretch(1)
            self._plot_right = None
            self._plot_bottom = None
            return

        self._plot_right = pg.PlotWidget()
        self._plot_right.setTitle("Left → Right (row v)")
        self._plot_right.setLabel("bottom", "X", units="m")
        self._plot_right.setLabel("left", "Height", units="m")
        self._plot_right.showGrid(x=True, y=True, alpha=0.3)
        self._curve_right = self._plot_right.plot(pen=pg.mkPen((0, 255, 239, 255), width=2))
        layout.addWidget(self._plot_right)

        self._plot_bottom = pg.PlotWidget()
        self._plot_bottom.setTitle("Top → Bottom (col u)")
        self._plot_bottom.setLabel("bottom", "Y", units="m")
        self._plot_bottom.setLabel("left", "Height", units="m")
        self._plot_bottom.showGrid(x=True, y=True, alpha=0.3)
        self._curve_bottom = self._plot_bottom.plot(pen=pg.mkPen((255, 200, 0, 255), width=2))
        layout.addWidget(self._plot_bottom)

    def set_cross_section(self, u: int, v: int, active: bool = True):
        """Update the cross-section location and refresh all linked views.

        Recomputes the two 1-D profile curves and moves the dashed slice lines
        on the 3D slab. Called both from the canvas crosshair drag (via
        label_widget) and from the toolbar spin-boxes.
        """
        if not active:
            self._cross_active = False
            self.gl_widget.clear_cross_slice()
            return
        self._cross_active = True
        h, w = self._depth_channel.shape[:2]
        u = min(max(int(u), 0), w - 1)
        v = min(max(int(v), 0), h - 1)
        self._cross_u, self._cross_v = u, v
        self._refresh_cross_section()

    def _compute_cross_section_state(self) -> dict:
        """Compute cross-section profiles for the current (u, v)."""
        return extract_cross_sections(
            self._depth_channel,
            self._cross_u,
            self._cross_v,
            depth_scale=self._depth_scale,
            field_width_m=self._field_width_m,
            field_height_m=self._field_height_m,
        )

    def _refresh_cross_section(self):
        """Recompute profiles and redraw curves + 3D slice lines."""
        if self._depth_channel is None or not self._cross_active:
            return
        cs = self._compute_cross_section_state()
        self._cross_zmin, self._cross_zmax = cs["zmin"], cs["zmax"]
        self._update_profile_curves(cs)
        self._update_slice_lines(cs)

    def _update_profile_curves(self, cs: dict):
        if not PYQTGRAPH_AVAILABLE:
            return
        # Use the GLOBAL depth range for both y-axes so relief is visible even
        # though the slab X/Y footprint spans ~3 m.
        ymin, ymax = cs["zmin"], cs["zmax"]
        pad = (ymax - ymin) * 0.1 + 1e-9
        if self._plot_right is not None:
            self._curve_right.setData(cs["left_right_x"], cs["left_right_z"])
            self._plot_right.setYRange(ymin - pad, ymax + pad)
            self._plot_right.setTitle(f"Left → Right (row {int(self._cross_v)})")
        if self._plot_bottom is not None:
            self._curve_bottom.setData(cs["top_bottom_y"], cs["top_bottom_z"])
            self._plot_bottom.setYRange(ymin - pad, ymax + pad)
            self._plot_bottom.setTitle(f"Top → Bottom (col {int(self._cross_u)})")

    def _update_slice_lines(self, cs: dict):
        z = float(cs["left_right_z"].max())
        x0 = float(cs["left_right_x"][self._cross_u])  # X at crosshair column
        y0 = float(cs["top_bottom_y"][self._cross_v])  # Y at crosshair row
        self.gl_widget.set_cross_slice(
            x0=x0,
            y0=y0,
            xmin=float(cs["left_right_x"][0]),
            xmax=float(cs["left_right_x"][-1]),
            ymin=float(cs["top_bottom_y"][0]),
            ymax=float(cs["top_bottom_y"][-1]),
            z=z,
        )

    def _on_toolbar_cross_section_changed(self, u: float, v: float):
        """Handle crosshair position change from the toolbar spin-boxes."""
        self.set_cross_section(u, v, True)
        # Write back to the canvas so the 2D cross matches the 3D spin-box.
        if (
            self._parent_widget is not None
            and hasattr(self._parent_widget, "_on_cross_section_from_3d")
        ):
            self._parent_widget._on_cross_section_from_3d(u, v)

    def _generate_3d_model(self):
        """Generate 3D model from selected channel."""
        if not OPEN3D_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, "Open3D Not Available",
                "Open3D is not installed. Please install with: pip install open3d>=0.19.0"
            )
            return

        # Select channel data
        if self._use_channel == "depth":
            depth_data = self._depth_channel
            color_data = None
            color_mode = "depth"
        elif self._use_channel == "reflectance":
            depth_data = self._reflectance_channel
            color_data = self._reflectance_channel
            color_mode = "reflectance"
        elif self._use_channel == "other":
            depth_data = self._other_channel
            color_data = self._other_channel
            color_mode = "other"
        else:
            depth_data = self._depth_channel
            color_data = None
            color_mode = "depth"

        if depth_data is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", f"No data available for channel: {self._use_channel}"
            )
            return

        # Estimate camera parameters
        fx = fy = self._focal_length
        if self._focal_length is None:
            fx = fy = None

        # Generate point cloud
        try:
            pcd = depth_to_pointcloud(
                depth=self._depth_channel,  # Use depth for geometry
                color=self._get_color_data(color_mode),  # Use selected channel for color
                depth_scale=self._depth_scale,  # Use depth scale for proper conversion
                fov_degrees=self._fov_degrees,  # Used when intrinsic params are None
                depth_is_range=self._depth_is_range,  # range vs perpendicular Z
                projection=self._projection,  # pinhole vs orthographic (top-view)
                field_width_m=self._field_width_m,  # physical footprint (orthographic)
                field_height_m=self._field_height_m,  # physical footprint (orthographic)
            )

            if self._current_pcd is not None:
                try:
                    self.gl_widget.view.removeItem(self.gl_widget._point_cloud_item)
                except ValueError:
                    # Item was already removed or not in the view
                    pass

            self._current_pcd = self._process_pointcloud(pcd)

            points, colors = self._pointcloud_to_arrays(self._current_pcd)

            if points is not None:
                self.gl_widget.set_point_cloud(points, colors)
                r2 = planarity_r2(points)
                self.status_bar.showMessage(
                    f"Points: {len(points)} · plane-fit R²={r2:.4f}"
                    if r2 == r2  # not NaN
                    else f"Points: {len(points)}"
                )
                if self._show_orientation and len(points) > 0:
                    z_base = float(points[:, 2].min())
                    z_max = float(points[:, 2].max())
                    # Raise labels slightly above the base so they float over the
                    # slab and are easier to read.
                    z_base = z_base + 0.05 * (z_max - z_base + 1e-6)
                    self.gl_widget.set_orientation_labels(
                        xmin=float(points[:, 0].min()),
                        xmax=float(points[:, 0].max()),
                        ymin=float(points[:, 1].min()),
                        ymax=float(points[:, 1].max()),
                        z_base=z_base,
                    )
                elif self._show_orientation:
                    self.gl_widget.clear_orientation()
            else:
                self.status_bar.showMessage("No points generated")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate 3D model: {e}")

    def _get_color_data(self, mode: str) -> Optional[np.ndarray]:
        """Get color data based on colorization mode."""
        if mode == "depth":
            # Convert depth to grayscale RGB for visualization
            if self._depth_channel is not None:
                # Normalize depth for visualization
                depth_norm = cv2.normalize(self._depth_channel, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_norm.astype(np.uint8)
                return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB)
            return None
        elif mode == "reflectance":
            if self._reflectance_channel is not None:
                ref_norm = cv2.normalize(self._reflectance_channel, None, 0, 255, cv2.NORM_MINMAX)
                ref_uint8 = ref_norm.astype(np.uint8)
                return cv2.cvtColor(ref_uint8, cv2.COLOR_GRAY2RGB)
            return None
        elif mode == "other":
            if self._other_channel is not None:
                other_norm = cv2.normalize(self._other_channel, None, 0, 255, cv2.NORM_MINMAX)
                other_uint8 = other_norm.astype(np.uint8)
                return cv2.cvtColor(other_uint8, cv2.COLOR_GRAY2RGB)
            return None
        return None

    def _process_pointcloud(self, pcd):
        """Process point cloud (downsample, remove outliers)."""
        from .pointcloud import downsample_pointcloud
        return downsample_pointcloud(pcd, voxel_size=0.005, remove_outliers=True)

    def _pointcloud_to_arrays(self, pcd) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        from .pointcloud import pointcloud_to_arrays
        return pointcloud_to_arrays(pcd)

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, self._on_reset_view)
        QtGui.QShortcut(QtGui.QKeySequence("F"), self, self._on_fit_view)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self, self.close)

    def _on_reset_view(self):
        self.gl_widget.reset_view()

    def _on_fit_view(self):
        self.gl_widget.fit_to_data()

    def _on_export(self, fmt: str):
        """Export point cloud or mesh."""
        if self._current_pcd is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No point cloud to export")
            return

        from .pointcloud import pointcloud_to_arrays
        points, colors = self._pointcloud_to_arrays(self._current_pcd)

        if points is None or len(points) == 0:
            QtWidgets.QMessageBox.warning(self, "No Data", "No points to export")
            return

        # Get save path
        fmt_lower = fmt.lower()
        filters = {
            "ply": "PLY Files (*.ply)",
            "obj": "OBJ Files (*.obj)",
            "npz": "NumPy Archive (*.npz)",
        }
        file_filter = filters.get(fmt.lower(), "All Files (*)")

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export as {fmt.upper()}", f"pointcloud.{fmt.lower()}", file_filter
        )

        if not file_path:
            return

        try:
            if fmt.lower() == "ply":
                self._export_ply(self._current_pcd, file_path)
            elif fmt.lower() == "obj":
                self._export_obj(file_path)
            elif fmt.lower() == "npz":
                points, colors = self.gl_widget._current_points, self.gl_widget._current_colors
                np.savez(file_path, points=points, colors=colors)
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported Format", f"Format {fmt} not supported")
                return

            self.status_bar.showMessage(f"Exported to {file_path}", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def _export_ply(self, pcd, file_path):
        import open3d as o3d
        o3d.io.write_point_cloud(file_path, pcd)

    def _export_obj(self, file_path):
        # For OBJ export, we'd need mesh - for now just export points as vertices
        points, colors = self.gl_widget._current_points, self.gl_widget._current_colors
        with open(file_path, 'w') as f:
            if points is not None:
                for i, pt in enumerate(points):
                    f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")

    def _on_point_size_changed(self, size: int):
        self.gl_widget.set_point_size(size)

    def _on_point_size_mode_changed(self, px_mode: bool):
        if self.gl_widget._point_cloud_item:
            self.gl_widget._point_cloud_item.setPxMode(px_mode)

    def _on_bg_color_changed(self, hex_color: str):
        self.gl_widget.set_background_color(QtGui.QColor(hex_color))

    def _on_grid_toggled(self, visible: bool):
        self.gl_widget.toggle_grid(visible)

    def _on_axis_toggled(self, visible: bool):
        self.gl_widget.toggle_axis(visible)

    def _on_orientation_toggled(self, visible: bool):
        self._show_orientation = bool(visible)
        self.gl_widget.toggle_orientation(bool(visible))

    def _on_projection_changed(self, perspective: bool):
        # GLViewWidget doesn't directly support ortho/perspective toggle
        # Would need custom implementation
        pass

    def _on_fov_changed(self, fov: float):
        """Handle FOV change from toolbar - store for the next render."""
        self._fov_degrees = float(fov)

    def _on_depth_is_range_changed(self, is_range: bool):
        """Handle depth-convention toggle - store for the next render."""
        self._depth_is_range = bool(is_range)

    def _on_reconstruction_mode_changed(self, mode: str):
        """Handle projection mode change (pinhole vs orthographic)."""
        self._projection = str(mode)

    def _on_field_size_changed(self, width_m: float, height_m: float):
        """Handle physical footprint change for the orthographic projection."""
        self._field_width_m = float(width_m)
        self._field_height_m = float(height_m)

    def _on_depth_scale_changed(self, scale: float):
        """Handle depth scale change from toolbar - store for the next render."""
        if scale == 0.0:  # Auto-detect
            self._depth_scale = self._auto_detect_depth_scale(self._depth_channel)
        else:
            self._depth_scale = scale

    def _on_render_requested(self):
        """Rebuild the point cloud with the current parameter settings."""
        self._generate_3d_model()
        if self._cross_active:
            self._refresh_cross_section()

    def _auto_detect_depth_scale(self, depth_channel: Optional[np.ndarray]) -> float:
        """Auto-detect depth scale from image statistics."""
        if depth_channel is None:
            return 255.0
        
        max_val = float(depth_channel.max())
        if max_val <= 1.0:
            return 1.0  # Already normalized
        elif max_val <= 255:
            return 255.0  # 8-bit: 0-255 -> 0-1 meters
        elif max_val <= 65535:
            return 1000.0  # 16-bit: assume millimeters
        return 1.0

    def _on_color_mode_changed(self, mode: str):
        """Handle color mode change - store for the next render."""
        self._color_mode = mode

    def _on_point_size_value_changed(self, size: int):
        self.gl_widget.set_point_size(size)

    def _on_mouse_moved(self, pos: QtCore.QPointF):
        """Handle mouse move for coordinate display."""
        # Could add coordinate display in status bar
        pass

    def closeEvent(self, event):
        """Handle window close; release the canvas crosshair if needed."""
        try:
            if (
                self._parent_widget is not None
                and hasattr(self._parent_widget, "_on_crosshair_view_closed")
            ):
                self._parent_widget._on_crosshair_view_closed()
        except Exception:
            pass
        super().closeEvent(event)