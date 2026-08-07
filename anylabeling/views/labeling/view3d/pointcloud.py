"""Depth map to point cloud conversion using Open3D."""

from typing import Optional, Tuple
import numpy as np
import cv2

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


def max_depth_meters_to_scale(
    max_depth_m: float, depth_max: float = 255.0
) -> float:
    """Convert a max real-world depth (meters) to a depth scale factor.

    Given an 8-bit depth channel spanning `0`..`depth_max`, a max real-world
    distance of `max_depth_m` meters yields `depth_scale = depth_max / max_depth_m`
    so that `depth_m = depth / depth_scale` spans `0`..`max_depth_m`.

    Args:
        max_depth_m: Maximum real-world depth in meters (must be > 0)
        depth_max: Maximum raw depth value (default 255 for 8-bit)

    Returns:
        The depth scale factor

    Raises:
        ValueError: If `max_depth_m` is zero or negative
    """
    if max_depth_m <= 0:
        raise ValueError("max_depth_m must be greater than 0")
    return float(depth_max) / float(max_depth_m)


def _normalize_color(
    color: np.ndarray, h: int, w: int
) -> np.ndarray:
    """Normalize an arbitrary color array to an HxWx3 uint8 RGB array.

    Open3D expects RGB u8 color (OpenCV color arrays are BGR). Supports u8,
    float 0-1, grayscale, or BGR/RGBA layouts.

    Args:
        color: Input color array
        h, w: Target height/width for the returned image

    Returns:
        HxWx3 uint8 RGB array
    """
    # Ensure color is uint8
    if color.dtype != np.uint8:
        if color.max() <= 1.0:
            color = (color * 255).astype(np.uint8)
        else:
            color = color.astype(np.uint8)

    # Convert to RGB format (Open3D expects RGB)
    if color.ndim == 3 and color.shape[2] == 3:
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    elif color.ndim == 3 and color.shape[2] == 4:
        color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
    elif color.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
    else:
        color = np.full((h, w, 3), 128, dtype=np.uint8)

    if color.shape[:2] != (h, w):
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)

    return color


def estimate_camera_params(
    image_shape: Tuple[int, int],
    fov_degrees: float = 60.0
) -> Tuple[float, float, float, float]:
    """Estimate camera intrinsics from image dimensions.

    Args:
        image_shape: (height, width) of the image
        fov_degrees: Field of view in degrees (default 60)

    Returns:
        (fx, fy, cx, cy) camera intrinsics
    """
    h, w = image_shape[:2]
    fov_rad = np.deg2rad(fov_degrees)
    fx = fy = (w / 2) / np.tan(np.deg2rad(fov_degrees) / 2)
    cx, cy = w / 2, h / 2
    return fx, fy, cx, cy


def depth_to_pointcloud(
    depth: np.ndarray,
    color: Optional[np.ndarray] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    depth_scale: float = 255.0,
    depth_trunc: float = 0.0,
    stride: int = 1,
    depth_min: float = 0.001,
    fov_degrees: float = 60.0,
    depth_is_range: bool = False,
    projection: str = "pinhole",
    field_width_m: float = 3.0,
    field_height_m: float = 3.0,
) -> Optional["o3d.geometry.PointCloud"]:
    """Convert depth map to Open3D point cloud.

    Args:
        depth: HxW depth map (normalized 0-1 or metric meters)
        color: Optional HxWx3 RGB uint8 image for rendering
        fx, fy: Focal lengths in pixels (auto-estimated if None)
        cx, cy: Principal point in pixels (default: image center)
        depth_scale: Scale factor to convert depth values to meters.
                     Default 255.0 for 8-bit (0-255 -> 0-1 meters).
                     Use 1000.0 for mm, 100.0 for cm, 1.0 for meters.
        depth_trunc: Maximum depth in meters (0 = no limit)
        stride: Downsampling stride (1 = full resolution)
        depth_min: Minimum depth in meters (filters out invalid/zero depth values)
        fov_degrees: Field of view in degrees used to auto-estimate intrinsics
                     when none are explicitly provided (default 60). Only used
                     for the "pinhole" projection.
        depth_is_range: Whether the depth values are Euclidean range (radial
                     distance along each pixel's ray) instead of perpendicular
                     Z-distance. When True, converts range to perpendicular Z
                     before pinhole projection, which keeps flat surfaces flat.
                     Only used for the "pinhole" projection.
        projection: "pinhole" (forward-looking camera via Open3D intrinsics) or
                     "orthographic" (top/nadir sensor where each pixel maps
                     directly to a ground cell and the value is height above
                     ground). For top-view data, orthographic avoids the
                     /Z coupling of pinhole that folds a flat field into a
                     pyramid. Default "pinhole".
        field_width_m: Physical width of the orthographic footprint in meters
                     (X axis). Default 3.0.
        field_height_m: Physical height of the orthographic footprint in meters
                     (Y axis). Default 3.0.

    Returns:
        Open3D PointCloud or None if Open3D not available
    """
    if not OPEN3D_AVAILABLE:
        return None

    h, w = depth.shape[:2]

    # Estimate camera intrinsics before downsampling (applied to the original
    # pixel grid) so pinhole reconstruction is stable regardless of stride.
    if fx is None or fy is None or cx is None or cy is None:
        est_fx, est_fy, est_cx, est_cy = estimate_camera_params(
            depth.shape, fov_degrees=fov_degrees
        )
        fx = fx if fx is not None else est_fx
        fy = fy if fy is not None else est_fy
        cx = cx if cx is not None else est_cx
        cy = cy if cy is not None else est_cy

    # Downsample if stride > 1
    if stride > 1:
        depth = depth[::stride, ::stride]
        if color is not None:
            color = color[::stride, ::stride]
        h, w = depth.shape[:2]

        # Ensure depth is in meters
    depth_m = depth.astype(np.float32) / depth_scale

    # Normalize the projection mode string (accept shorthand/case variants).
    proj = (projection or "pinhole").strip().lower()

    if proj == "orthographic":
        # Top/nadir sensor: each pixel is a ground cell. Map (u, v) directly to
        # (X, Y) meters using the physical field footprint, and treat the depth
        # value as height above ground (Z). The constant-128 padding ring in
        # real data then reconstructs as a flat rectangular slab instead of a
        # pyramid.
        if field_width_m <= 0 or field_height_m <= 0:
            raise ValueError("field_width_m and field_height_m must be greater than 0")
        cell_x = float(field_width_m) / w
        cell_y = float(field_height_m) / h
        ys, xs = np.mgrid[0:h, 0:w]
        x = ((xs - (w - 1) / 2.0) * cell_x).astype(np.float32)
        y = ((ys - (h - 1) / 2.0) * cell_y).astype(np.float32)

        if depth_min > 0:
            depth_m = np.where(depth_m >= depth_min, depth_m, 0.0)
        if depth_trunc > 0:
            depth_m = np.where(depth_m <= depth_trunc, depth_m, 0.0)

        valid = depth_m > 0
        points = np.column_stack(
            [x[valid].ravel(), y[valid].ravel(), depth_m[valid].ravel()]
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if color is not None:
            color = _normalize_color(color, h, w)
            pcd.colors = o3d.utility.Vector3dVector(
                (color[valid].reshape(-1, 3) / 255.0).astype(np.float32)
            )
        return pcd

    # --- Pinhole (forward-looking camera) path below ---

    # Convert Euclidean range to perpendicular Z so a flat surface stays flat.
    # For a pixel with normalized ray offsets du, dv, the perpendicular depth is
    # Z = range / sqrt(1 + du^2 + dv^2). Without this, feeding range as Z
    # inflates the corners and folds a flat road into a pyramid.
    if depth_is_range:
        ys, xs = np.mgrid[0:h, 0:w]
        du = (xs - cx) / fx
        dv = (ys - cy) / fy
        depth_m = depth_m / np.sqrt(1.0 + du * du + dv * dv)
        depth_m = depth_m.astype(np.float32)
    
    # Filter out invalid depth values (too close or zero)
    if depth_min > 0:
        depth_m = np.where(depth_m >= depth_min, depth_m, 0.0)
    
    # Apply depth truncation
    if depth_trunc > 0:
        depth_m = np.where(depth_m <= depth_trunc, depth_m, 0)

    # Open3D's RGBD constructor treats depth_trunc=0 as "truncate everything",
    # while the depth-only path treats 0 as "no limit". Use an effective
    # truncation so depth_trunc=0 means "no limit" in both paths.
    effective_trunc = depth_trunc if depth_trunc > 0 else float(depth_m.max() * 2.0 + 1e-6)

    # Create Open3D camera intrinsics
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=int(w), height=int(h), fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Convert depth to Open3D Image
    depth_o3d = o3d.geometry.Image(depth_m)

    if color is not None:
        color = _normalize_color(color, h, w)

        color_o3d = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            o3d.geometry.Image(depth_m),
            depth_scale=1.0,
            depth_trunc=effective_trunc,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                width=int(w), height=int(h), fx=fx, fy=fy, cx=cx, cy=cy
            ),
        )
    else:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            o3d.camera.PinholeCameraIntrinsic(
                width=int(w), height=int(h), fx=fx, fy=fy, cx=cx, cy=cy
            ),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
        )

    return pcd


def extract_cross_sections(
    depth: np.ndarray,
    u: float,
    v: float,
    depth_scale: float = 255.0,
    field_width_m: float = 3.0,
    field_height_m: float = 3.0,
) -> dict:
    """Extract 1-D depth cross-section profiles along the horizontal and
    vertical image axes through the crosshair position (u, v).

    Matches the orthographic mapping used by :func:`depth_to_pointcloud`:
    ``X = (u - (w-1)/2) * cell_x``, ``Y = (v - (h-1)/2) * cell_y`` and
    ``Z = depth / depth_scale``.

    Args:
        depth: HxW depth map (raw values, before scaling)
        u: Crosshair column (image x) in pixels
        v: Crosshair row (image y) in pixels
        depth_scale: Scale factor converting raw depth to meters (Z axis)
        field_width_m: Physical footprint width in meters (X axis)
        field_height_m: Physical footprint height in meters (Y axis)

    Returns:
        dict with keys:
            left_right_x: shape (W,) X positions (m) along the row at ``v``
            left_right_z: shape (W,) Z heights (m) at ``depth[v, :]``
            top_bottom_y: shape (H,) Y positions (m) along the column at ``u``
            top_bottom_z: shape (H,) Z heights (m) at ``depth[:, u]``
            zmin, zmax: global valid depth range in meters (y-axis limits)
            u, v:	clamped integer crosshair column/row
    """
    h, w = depth.shape[:2]
    cell_x = float(field_width_m) / w
    cell_y = float(field_height_m) / h
    depth_m = depth.astype(np.float32) / depth_scale

    row = int(round(v))
    row = min(max(row, 0), h - 1)
    col = int(round(u))
    col = min(max(col, 0), w - 1)

    xs = (np.arange(w) - (w - 1) / 2.0) * cell_x
    ys = (np.arange(h) - (h - 1) / 2.0) * cell_y

    valid = depth_m > 0
    if valid.any():
        zmin = float(depth_m[valid].min())
        zmax = float(depth_m[valid].max())
    else:
        zmin = zmax = 0.0

    return {
        "left_right_x": xs.astype(np.float32),
        "left_right_z": depth_m[row, :].astype(np.float32),
        "top_bottom_y": ys.astype(np.float32),
        "top_bottom_z": depth_m[:, col].astype(np.float32),
        "zmin": zmin,
        "zmax": zmax,
        "u": float(col),
        "v": float(row),
    }


def downsample_pointcloud(
    pcd: "o3d.geometry.PointCloud",
    voxel_size: float = 0.01,
    remove_outliers: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> "o3d.geometry.PointCloud":
    """Downsample and clean point cloud.

    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling (in meters)
        remove_outliers: Whether to remove statistical outliers
        nb_neighbors: Number of neighbors for outlier removal
        std_ratio: Standard deviation ratio for outlier removal

    Returns:
        Processed point cloud
    """
    if not OPEN3D_AVAILABLE:
        return None

    # Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Remove statistical outliers
    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

    return pcd


def pointcloud_to_arrays(pcd: "o3d.geometry.PointCloud") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert Open3D point cloud to numpy arrays.

    Args:
        pcd: Open3D point cloud

    Args:
        pcd: Open3D point cloud

    Returns:
        (points, colors) where points is Nx3 float32, colors is Nx3 uint8 or None
    """
    if not OPEN3D_AVAILABLE:
        return None, None

    points = np.asarray(pcd.points, dtype=np.float32)

    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        colors = None

    return points, colors


def planarity_r2(points: np.ndarray) -> float:
    """Return the least-squares plane-fit R^2 of a point cloud.

    Fits ``z = a*x + b*y + c`` to the points and returns the coefficient of
    determination (1 = perfectly planar). This is used to verify that a depth
    image of a flat surface reconstructs as a flat plane (R^2 near 1) with the
    chosen camera intrinsics.

    Args:
        points: Nx3 float array of (x, y, z) points

    Returns:
        R^2 in [0, 1] (or NaN for degenerate/empty input)
    """
    if points is None or len(points) < 3:
        return float("nan")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    a = np.column_stack([x, y, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
    z_pred = a @ coef
    ss_res = float(np.sum((z - z_pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    if ss_tot == 0:
        # Perfect constant-plane fit (no variance in z): report as planar.
        # A small epsilon tolerates float rounding in the reconstruction.
        return 1.0 if ss_res < 1e-12 else float("nan")
    return 1.0 - ss_res / ss_tot


def create_pointcloud_from_depth_image(
    depth_image: np.ndarray,
    color_image: Optional[np.ndarray] = None,
    focal_length: Optional[float] = None,
    depth_scale: float = 255.0,
    max_points: int = 100000,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """High-level function to create point cloud arrays from depth image.

    This is a convenience function that handles the full pipeline.

    Args:
        depth_image: HxW depth map
        color_image: Optional HxWx3 RGB image
        focal_length: Focal length in pixels (auto-estimated if None)
        depth_scale: Scale factor for depth values (default 255 for 8-bit)
        max_points: Maximum number of points (downsamples if exceeded)

    Returns:
        (points, colors) arrays for use with PyQtGraph
    """
    if not OPEN3D_AVAILABLE:
        return None, None

    pcd = depth_to_pointcloud(
        depth=depth_image,
        color=color_image,
        depth_scale=depth_scale,
    )

    if pcd is None:
        return None, None

    # Downsample if too many points
    num_points = len(pcd.points)
    if num_points > max_points:
        voxel_size = 0.005 * (num_points / max_points) ** (1/3)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return pointcloud_to_arrays(pcd)