"""PyQtGraph GLViewWidget wrapper for 3D visualization."""

import numpy as np
from typing import Optional, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph.opengl as gl
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    gl = None


class GLViewerWidget(QtWidgets.QWidget):
    """PyQtGraph GLViewWidget embedded in QWidget for PyQt6."""

    # Signals
    mouse_moved = QtCore.pyqtSignal(QtCore.QPointF)
    point_selected = QtCore.pyqtSignal(int, QtCore.QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)

        if not PYQTGRAPH_AVAILABLE:
            raise ImportError("pyqtgraph is required for 3D visualization")

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Create GLViewWidget
        self.view = gl.GLViewWidget()
        self._layout.addWidget(self.view)

        # Initialize view
        self._init_view()

        # State
        self._point_cloud_item = None
        self._grid_item = None
        self._axis_item = None
        self._current_points = None
        self._current_colors = None
        self._orientation_items = []
        self._cross_items = []

        # Enable mouse tracking for hover effects
        self.view.setMouseTracking(True)
        self.view.installEventFilter(self)

    def _init_view(self):
        """Initialize the 3D view with default settings."""
        # Set background color
        self.view.setBackgroundColor(QtGui.QColor(30, 30, 30))

        # Add grid
        self._grid_item = gl.GLGridItem()
        self._grid_item.setSize(x=10, y=10, z=10)
        self._grid_item.setSpacing(x=1, y=1, z=1)
        self.view.addItem(self._grid_item)

        # Add axis
        self._axis_item = gl.GLAxisItem()
        self._axis_item.setSize(x=1, y=1, z=1)
        self.view.addItem(self._axis_item)

        # Set initial camera position
        self.view.setCameraPosition(distance=2, elevation=30, azimuth=45)

        # Set up lighting
        self.view.setBackgroundColor(QtGui.QColor(30, 30, 30))

    def set_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_size: float = 1.0
    ) -> None:
        """Set point cloud data.

        Args:
            points: Nx3 float32 array of point coordinates
            colors: Optional Nx3 uint8 array of RGB colors
            point_size: Size of points in pixels
        """
        if len(points) == 0:
            self.clear()
            return

        # Remove existing point cloud
        if self._point_cloud_item is not None:
            try:
                self.view.removeItem(self._point_cloud_item)
            except ValueError:
                # Item was already removed or not in the view
                pass

        # Create scatter plot item
        self._point_cloud_item = gl.GLScatterPlotItem(
            pos=points,
            color=colors / 255.0 if colors is not None else None,
            size=point_size,
            pxMode=True,  # Size in pixels
        )
        self.view.addItem(self._point_cloud_item)

        self._current_points = points
        self._current_colors = colors

        # Auto-fit view to point cloud
        self.fit_to_data()

    def set_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    ) -> None:
        """Set mesh data.

        Args:
            vertices: Nx3 float32 array of vertex coordinates
            faces: Mx3 int32 array of face indices
            color: RGBA color tuple (0-1 range)
        """
        if len(vertices) == 0:
            return

        # Remove existing mesh items
        for item in list(self.view.items):
            if isinstance(item, gl.GLMeshItem):
                try:
                    self.view.removeItem(item)
                except ValueError:
                    # Item was already removed or not in the view
                    pass

        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        mesh_item = gl.GLMeshItem(
            meshdata=mesh_data,
            color=color,
            smooth=True,
            shader="shaded",
            glOptions="opaque",
        )
        self.view.addItem(mesh_item)

    def clear(self) -> None:
        """Clear all point clouds and meshes."""
        if self._point_cloud_item is not None:
            try:
                self.view.removeItem(self._point_cloud_item)
            except ValueError:
                # Item was already removed or not in the view
                pass
            self._point_cloud_item = None
        self._current_points = None
        self._current_colors = None
        self.clear_orientation()
        self.clear_cross_slice()

    @staticmethod
    def _dashed_points(
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        dash: float = 0.05,
        gap: float = 0.04,
    ) -> np.ndarray:
        """Generate a dashed line as a sequence of disconnected segments.

        Returns an (N, 3) array where consecutive pairs describe individual
        line segments (GL 'lines' mode) so the result reads as a dashed stroke.
        """
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        length = float(np.linalg.norm(p1 - p0))
        if length <= 0:
            return np.empty((0, 3), dtype=np.float32)
        step = max(dash + gap, 1e-6)
        dirn = (p1 - p0) / length
        segments = []
        dist = 0.0
        while dist < length:
            s = min(dist + dash, length)
            segments.append(p0 + dirn * dist)
            segments.append(p0 + dirn * s)
            dist += step
        return np.asarray(segments, dtype=np.float32)

    def set_cross_slice(
        self,
        x0: float,
        y0: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        z: float = 0.0,
    ) -> None:
        """Draw dashed slice lines on the slab at the crosshair position.

        Two perpendicular dashed lines on the top surface: one along the X axis
        at ``y0`` (the Left-Right profile location) and one along the Y axis at
        ``x0`` (the Top-Bottom profile location).

        Args:
            x0, y0: Ground (X, Y) of the crosshair intersection
            xmin, xmax: X extent of the footprint
            ymin, ymax: Y extent of the footprint
            z: Height at which the lines are drawn (slab top)
        """
        self.clear_cross_slice()
        color = (0, 255, 239, 255)  # cyan, opaque
        span_x = float(xmax) - float(xmin)
        span_y = float(ymax) - float(ymin)

        # Left-to-right: constant Y = y0, X varies across the footprint.
        if span_x > 1e-9:
            pts = self._dashed_points(
                (float(xmin), float(y0), float(z)),
                (float(xmax), float(y0), float(z)),
                dash=0.08 * span_x,
                gap=0.05 * span_x,
            )
            if len(pts):
                item = gl.GLLinePlotItem(pos=pts, color=color, width=2.0, mode="lines")
                self.view.addItem(item)
                self._cross_items.append(item)

        # Top-to-bottom: constant X at x0, Y varies across Y range.
        if span_y > 1e-9:
            pts = self._dashed_points(
                (x0, float(ymin), float(z)),
                (x0, float(ymax), float(z)),
                dash=0.08 * span_y,
                gap=0.05 * span_y,
            )
            if len(pts):
                item = gl.GLLinePlotItem(pos=pts, color=color, width=2.0, mode="lines")
                self.view.addItem(item)
                self._cross_items.append(item)

    def clear_cross_slice(self) -> None:
        """Remove the cross-section slice lines from the view."""
        for item in self._cross_items:
            try:
                self.view.removeItem(item)
            except ValueError:
                pass
        self._cross_items = []

    def clear_orientation(self) -> None:
        """Remove all image-direction labels from the view."""
        for item in self._orientation_items:
            try:
                self.view.removeItem(item)
            except ValueError:
                # Item was already removed or not in the view
                pass
        self._orientation_items = []

    def set_orientation_labels(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        z_base: float = 0.0,
    ) -> None:
        """Add image-direction edge labels (Left/Right/Top/Bottom).

        Anchors each label to the midpoint of the corresponding edge of the
        model's X/Y footprint, raised slightly above the slab height so the
        text stays readable above the points.

        Args:
            xmin, xmax: X extent of the reconstructed footprint
            ymin, ymax: Y extent of the reconstructed footprint
            z_base: Z (height) at which the labels are placed
        """
        self.clear_orientation()
        z = float(z_base)

        span_x = float(xmax) - float(xmin)
        span_y = float(ymax) - float(ymin)
        pad = 0.10 * max(span_x, span_y)

        label_color = (255, 215, 0, 255)  # gold, opaque (0-255 ints)
        label_font = QtGui.QFont("Helvetica", 36)

        def edge(pos, text):
            item = gl.GLTextItem(
                pos=np.asarray(pos, dtype=np.float32),
                text=text,
                color=label_color,
                font=label_font,
            )
            self.view.addItem(item)
            self._orientation_items.append(item)
            return item

        # Place labels just outside the slab edges so points never occlude them.
        edge((xmin - pad, 0, z), "Image Left")
        edge((xmax + pad, 0, z), "Image Right")
        edge((0, ymin - pad, z), "Image Top")
        edge((0, ymax + pad, z), "Image Bottom")

    def toggle_orientation(self, visible: bool) -> None:
        """Show/hide orientation labels."""
        for item in self._orientation_items:
            item.setVisible(bool(visible))

    def fit_to_data(self, margin: float = 1.1) -> None:
        """Fit camera view to encompass all data."""
        if self._current_points is None or len(self._current_points) == 0:
            return

        points = self._current_points
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        center = (min_vals + max_vals) / 2
        extent = (max_vals - min_vals) * margin

        # Set camera to view all data
        max_extent = max(extent)
        distance = max_extent * 1.5

        self.view.setCameraPosition(
            distance=max(distance, 0.1),
            elevation=30,
            azimuth=45,
            pos=QtGui.QVector3D(*center)
        )

    def reset_view(self) -> None:
        """Reset camera to default position."""
        self.view.setCameraPosition(distance=2, elevation=30, azimuth=45)

    def set_point_size(self, size: float) -> None:
        """Update point size for point cloud."""
        if self._point_cloud_item is not None:
            self._point_cloud_item.setData(size=size)

    def set_background_color(self, color: QtGui.QColor) -> None:
        """Set background color."""
        self.view.setBackgroundColor(color)

    def toggle_grid(self, visible: bool) -> None:
        """Show/hide grid."""
        if self._grid_item:
            self._grid_item.setVisible(visible)

    def toggle_axis(self, visible: bool) -> None:
        """Show/hide axis."""
        if self._axis_item:
            self._axis_item.setVisible(visible)

    def get_camera_params(self) -> dict:
        """Get current camera parameters."""
        # Note: GLViewWidget doesn't expose camera params directly
        # This would need custom implementation
        return {
            "distance": self.view.opts.get("distance", 2),
            "elevation": self._get_elevation(),
            "azimuth": self._get_azimuth(),
            "center": self._get_center(),
        }

    def _get_elevation(self) -> float:
        """Get camera elevation angle."""
        # Not directly exposed by GLViewWidget
        return 30.0

    def _get_azimuth(self) -> float:
        """Get camera azimuth angle."""
        return 45.0

    def _get_center(self) -> tuple:
        """Get camera center point."""
        return (0.0, 0.0, 0.0)

    def eventFilter(self, obj, event):
        """Filter events for mouse tracking."""
        if obj is self.view:
            if event.type() == QtCore.QEvent.Type.MouseMove:
                self.mouse_moved.emit(event.position())
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press for point selection."""
        super().mousePressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Handle wheel zoom."""
        super().wheelEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == QtCore.Qt.Key.Key_R:
            self.reset_view()
        elif event.key() == QtCore.Qt.Key.Key_F:
            self.fit_to_data()
        elif event.key() == QtCore.Qt.Key.Key_Plus or event.key() == QtCore.Qt.Key.Key_Equal:
            self.view.opts["distance"] *= 0.9
        elif event.key() == QtCore.Qt.Key.Key_Minus:
            self.view.opts["distance"] *= 1.1
        else:
            super().keyPressEvent(event)


from typing import Optional, Tuple