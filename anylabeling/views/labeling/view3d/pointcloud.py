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
) -> Optional["o3d.geometry.PointCloud"]:
    """Convert depth map to Open3D point cloud.

    Args:
        depth: HxW depth map (normalized 0-1 or metric meters)
        color: Optional HxWx3 RGB uint8 image for coloring
        fx, fy: Focal lengths in pixels (auto-estimated if None)
        cx, cy: Principal point in pixels (default: image center)
        depth_scale: Scale factor to convert depth values to meters.
                     Default 255.0 for 8-bit (0-255 -> 0-1 meters).
                     Use 1000.0 for mm, 100.0 for cm, 1.0 for meters.
        depth_trunc: Maximum depth in meters (0 = no limit)
        stride: Downsampling stride (1 = full resolution)
        depth_min: Minimum depth in meters (filters out invalid/zero depth values)

    Returns:
        Open3D PointCloud or None if Open3D not available
    """
    if not OPEN3D_AVAILABLE:
        return None

    h, w = depth.shape[:2]

    # Estimate camera intrinsics if not provided
    if fx is None or fy is None or cx is None or cy is None:
        est_fx, est_fy, est_cx, est_cy = estimate_camera_params(depth.shape)
        fx = fx or est_fx
        fy = fy or est_fy
        cx = cx or est_cx
        cy = cy or est_cy

    # Downsample if stride > 1
    if stride > 1:
        depth = depth[::stride, ::stride]
        if color is not None:
            color = color[::stride, ::stride]
        h, w = depth.shape[:2]

# Ensure depth is in meters
    depth_m = depth.astype(np.float32) / depth_scale
    
    # Filter out invalid depth values (too close or zero)
    if depth_min > 0:
        depth_m = np.where(depth_m >= depth_min, depth_m, 0.0)
    
    # Apply depth truncation
    if depth_trunc > 0:
        depth_m = np.where(depth_m <= depth_trunc, depth_m, 0)
    
    # Create Open3D camera intrinsics
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=int(w), height=int(h), fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Convert depth to Open3D Image
    depth_o3d = o3d.geometry.Image(depth_m)

    if color is not None:
        # Ensure color is uint8
        if color.dtype != np.uint8:
            if color.max() <= 1.0:
                color = (color * 255).astype(np.uint8)
            else:
                color = color.astype(np.uint8)

        # Fix: Convert to RGB format (Open3D expects RGB)
        if color.ndim == 3 and color.shape[2] == 3:
            # Convert BGR to RGB (OpenCV uses BGR, Open3D expects RGB)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        elif color.ndim == 3 and color.shape[2] == 4:
            # Handle RGBA -> RGB
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
        elif color.ndim == 2:
            # Grayscale -> RGB
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
        else:
            # Unexpected format, create a default gray image
            h, w = color.shape[:2] if color.ndim >= 2 else (480, 640)
            color = np.full((h, w, 3), 128, dtype=np.uint8)

        color_o3d = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            o3d.geometry.Image(depth_m),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
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