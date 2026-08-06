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