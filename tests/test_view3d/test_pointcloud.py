import unittest

import numpy as np

from anylabeling.views.labeling.view3d.pointcloud import (
    OPEN3D_AVAILABLE,
    depth_to_pointcloud,
    max_depth_meters_to_scale,
    planarity_r2,
)


def _make_depth_image(h=64, w=64):
    """Create a synthetic depth map with a realistic 8-bit range."""
    # Depth values span a gradient, all above the depth_min threshold.
    depth = np.tile(np.linspace(5, 255, w, dtype=np.float32), (h, 1))
    return depth


def _make_color_image(depth):
    h, w = depth.shape[:2]
    color = np.full((h, w, 3), 200, dtype=np.uint8)
    return color


class TestMaxDepthMetersToScale(unittest.TestCase):
    def test_scale_for_road_depth(self):
        self.assertAlmostEqual(max_depth_meters_to_scale(0.5), 510.0)
        self.assertAlmostEqual(max_depth_meters_to_scale(0.3), 850.0)
        self.assertAlmostEqual(max_depth_meters_to_scale(1.0), 255.0)

    def test_custom_depth_max(self):
        self.assertAlmostEqual(max_depth_meters_to_scale(2.0, depth_max=1024), 512.0)

    def test_zero_or_negative_raises(self):
        with self.assertRaises(ValueError):
            max_depth_meters_to_scale(0.0)
        with self.assertRaises(ValueError):
            max_depth_meters_to_scale(-0.1)


@unittest.skipUnless(OPEN3D_AVAILABLE, "Open3D is required for 3D view tests")
class TestDepthToPointcloud(unittest.TestCase):
    def test_rgbd_path_default_depth_trunc_keeps_points(self):
        """The RGBD path must produce points with the default
        ``depth_trunc=0.0``. Open3D's RGBD constructor truncates everything at
        depth_trunc=0, which used to zero out every depth pixel (no model
        shown, "Points: 0")."""
        depth = _make_depth_image()
        color = _make_color_image(depth)

        pcd = depth_to_pointcloud(depth=depth, color=color, depth_scale=255.0)

        self.assertIsNotNone(pcd)
        self.assertGreater(len(pcd.points), 0)

    def test_depth_only_path_default_depth_trunc_keeps_points(self):
        """Regression guard: the depth-only path already worked with
        ``depth_trunc=0`` and must keep working."""
        depth = _make_depth_image()

        pcd = depth_to_pointcloud(depth=depth, color=None, depth_scale=255.0)

        self.assertIsNotNone(pcd)
        self.assertGreater(len(pcd.points), 0)

    def test_rgbd_path_respects_positive_depth_trunc(self):
        """A positive ``depth_trunc`` must still be honored: points beyond the
        truncation distance are removed."""
        depth = _make_depth_image()
        color = _make_color_image(depth)

        pcd = depth_to_pointcloud(
            depth=depth,
            color=color,
            depth_scale=255.0,
            depth_trunc=0.1,
        )

        self.assertIsNotNone(pcd)
        self.assertGreater(len(pcd.points), 0)
        self.assertLess(len(pcd.points), depth.size)


@unittest.skipUnless(OPEN3D_AVAILABLE, "Open3D is required for 3D view tests")
class TestPlanarityAndFov(unittest.TestCase):
    def _flat_cloud(self):
        depth = np.full((200, 64), 5.0, dtype=np.float32)
        color = _make_color_image(depth)
        pcd = depth_to_pointcloud(depth=depth, color=color, depth_scale=255.0)
        return np.asarray(pcd.points)

    def test_constant_depth_is_planar(self):
        """A constant-depth (fronto-parallel) surface must reconstruct as a
        flat plane (plane-fit R² near 1)."""
        self.assertAlmostEqual(planarity_r2(self._flat_cloud()), 1.0, places=3)

    def test_non_planar_funnel_is_detected(self):
        """A radial funnel must be detected as non-planar (R² ≪ 1)."""
        yy, xx = np.mgrid[0:200, 0:64]
        r = np.sqrt((xx - 32) ** 2 + (yy - 100) ** 2)
        depth = (10.0 - np.clip(r / 20.0, 0, 10)).astype(np.float32)
        color = _make_color_image(depth)
        pcd = depth_to_pointcloud(depth=depth, color=color, depth_scale=255.0)
        self.assertLess(planarity_r2(np.asarray(pcd.points)), 0.5)

    def test_fov_changes_lateral_extent(self):
        """Higher FOV lowers the auto-estimated focal length, which spreads
        points laterally wider for the same depth."""
        depth = np.full((200, 64), 5.0, dtype=np.float32)
        color = _make_color_image(depth)
        narrow = depth_to_pointcloud(
            depth=depth, color=color, depth_scale=255.0, fov_degrees=60.0
        )
        wide = depth_to_pointcloud(
            depth=depth, color=color, depth_scale=255.0, fov_degrees=140.0
        )
        span_n = np.asarray(narrow.points)[:, 0].max() - np.asarray(narrow.points)[:, 0].min()
        span_w = np.asarray(wide.points)[:, 0].max() - np.asarray(wide.points)[:, 0].min()
        self.assertGreater(span_w, span_n)

    def _tilted_plane(self):
        """Build a flat tilted plane as both perpendicular-Z and Euclidean
        range. The perpendicular field z = c/(1 - a*du - b*dv) varies across the
        image so R² is a meaningful (non-degenerate) planarity measure."""
        h, w = 200, 64
        fx = fy = (w / 2) / np.tan(np.deg2rad(60) / 2)
        cx, cy = w / 2, h / 2
        ys, xs = np.mgrid[0:h, 0:w]
        du = (xs - cx) / fx
        dv = (ys - cy) / fy
        a, b, c = 0.2, -0.1, 5.0
        z = (c / (1.0 - a * du - b * dv)).astype(np.float32)
        rng = (z * np.sqrt(1.0 + du * du + dv * dv)).astype(np.float32)
        color = _make_color_image(z)
        return z, rng, color

    def test_range_mislabeled_as_perp_folds_into_pyramid(self):
        """Regression guard: feeding Euclidean range as perpendicular Z folds a
        flat plane toward a pyramid (lower planarity than the true plane)."""
        _, rng, color = self._tilted_plane()
        pcd = depth_to_pointcloud(
            depth=rng, color=color, depth_scale=1.0, depth_is_range=False
        )
        self.assertLess(planarity_r2(np.asarray(pcd.points)), 0.8)

    def test_range_converted_to_perp_is_planar(self):
        """With depth_is_range=True, a flat plane's range must reconstruct as
        planar (R² near 1), matching the true perpendicular plane (this fixes
        the pyramid)."""
        z, rng, color = self._tilted_plane()
        true_pcd = depth_to_pointcloud(depth=z, color=color, depth_scale=1.0)
        rng_pcd = depth_to_pointcloud(
            depth=rng, color=color, depth_scale=1.0, depth_is_range=True
        )
        self.assertGreater(planarity_r2(np.asarray(true_pcd.points)), 0.99)
        self.assertGreater(planarity_r2(np.asarray(rng_pcd.points)), 0.99)


@unittest.skipUnless(OPEN3D_AVAILABLE, "Open3D is required for 3D view tests")
class TestOrthographicProjection(unittest.TestCase):
    """Top/nadir sensor reconstruction: each pixel maps to a ground cell and
    the depth value is height above ground."""

    def _ortho_cloud(self, depth, field_w=3.0, field_h=3.0, color=None):
        pcd = depth_to_pointcloud(
            depth=depth.astype(np.float32),
            color=color,
            depth_scale=1.0,
            projection="orthographic",
            field_width_m=field_w,
            field_height_m=field_h,
        )
        return np.asarray(pcd.points)

    def _span(self, pts, axis):
        return float(pta.max() - pta.min()) if (pta := pts[:, axis]).size else 0.0

    def test_constant_height_is_planar(self):
        """A constant-height top-view field must reconstruct as a flat plane
        (the road slab, not a pyramid)."""
        depth = np.full((200, 200), 0.5, dtype=np.float32)
        pts = self._ortho_cloud(depth)
        self.assertGreater(planarity_r2(pts), 0.99)
        self.assertEqual(len(pts), depth.size)

    def test_height_variance_is_preserved_as_z(self):
        """Brighter = higher: the depth value maps directly to Z, so a ramp in
        the image becomes a plane tilted in Z with R² near 1."""
        yy, xx = np.mgrid[0:128, 0:128]
        depth = (0.1 + 0.3 * xx / 127.0).astype(np.float32)
        pts = self._ortho_cloud(depth)
        self.assertGreater(planarity_r2(pts), 0.99)
        z = pts[:, 2]
        self.assertAlmostEqual(z.min(), 0.1, places=3)
        self.assertAlmostEqual(z.max(), 0.4, places=3)

    def test_orthographic_is_rectangular_not_pyramid(self):
        """The reconstructed cloud must span a rectangle in X/Y (a prism) even
        with an asymmetric height field. A pinhole projection of the same field
        folds corners inward (cone-like) and reduces planarity."""
        depth = np.full((200, 200), 0.5, dtype=np.float32)
        # asymmetric bulge so pinhole is pushed into a cone while orthographic
        # stays a clean rectangle
        depth[50:150, 50:150] = 0.9
        pts = self._ortho_cloud(depth)
        # width spans 3 m: cell = 3/200, so span = (200-1)*3/200
        expected = (200 - 1) * 3.0 / 200.0
        self.assertAlmostEqual(self._span(pts, 0), expected, places=3)
        self.assertAlmostEqual(self._span(pts, 1), expected, places=3)

    def test_field_size_sets_footprint(self):
        """The physical field size must set the X/Y footprint exactly; a 3 m
        field on a ramp height keeps planarity."""
        yy, xx = np.mgrid[0:128, 0:128]
        depth = (0.1 + 0.3 * yy / 127.0).astype(np.float32)
        pts = self._ortho_cloud(depth, field_w=3.0, field_h=3.0)
        w = (128 - 1) * 3.0 / 128.0
        self.assertAlmostEqual(self._span(pts, 0), w, places=3)
        self.assertAlmostEqual(self._span(pts, 1), w, places=3)
        self.assertGreater(planarity_r2(pts), 0.99)

    def test_field_size_scales_lateral_extent(self):
        """A smaller field (0.03 m) shrinks the footprint ~100x vs 3 m."""
        depth = np.full((200, 200), 0.5, dtype=np.float32)
        wide = self._ortho_cloud(depth, field_w=3.0, field_h=3.0)
        narrow = self._ortho_cloud(depth, field_w=0.03, field_h=0.03)
        span_w = self._span(wide, 0)
        span_n = self._span(narrow, 0)
        self.assertAlmostEqual(span_w / span_n, 100.0, places=2)

    def test_orthographic_preserves_height_range(self):
        """Z (height) must equal the image value range regardless of the field
        size - no vertical exaggeration is applied."""
        depth = np.full((128, 128), 0.25, dtype=np.float32)
        depth[30:98, 30:98] = 0.85

        def zrange(fw):
            pts = self._ortho_cloud(depth, field_w=fw, field_h=fw)
            return pts[:, 2].max() - pts[:, 2].min()

        self.assertAlmostEqual(zrange(3.0), 0.60, places=3)
        self.assertAlmostEqual(zrange(0.03), 0.60, places=3)

    def test_padding_ring_keeps_flat_footprint(self):
        """Regression guard matching real top-view data: a flat ``128`` padding
        ring around a slightly higher interior must still reconstruct with a
        rectangular ground footprint (a flat slab), not a pyramid that pulls
        the corners inward."""
        depth = np.full((256, 256), 0.2, dtype=np.float32)
        # constant grey padding ring (simulated 128) around a flat interior
        depth[20:-20, 20:-20] = 0.5
        depth[40:-40, 40:-40] = 0.75
        pts = self._ortho_cloud(depth, field_w=3.0, field_h=3.0)
        w = (256 - 1) * 3.0 / 256.0
        self.assertAlmostEqual(self._span(pts, 0), w, places=3)
        self.assertAlmostEqual(self._span(pts, 1), w, places=3)
        # Z is height: every point's Z equals one of the laid-back heights
        self.assertSetEqual(
            set(np.unique(np.round(pts[:, 2], 2))), {0.2, 0.5, 0.75}
        )

    def test_zero_field_size_raises(self):
        with self.assertRaises(ValueError):
            self._ortho_cloud(np.full((16, 16), 0.5), field_w=0.0, field_h=3.0)
        with self.assertRaises(ValueError):
            self._ortho_cloud(np.full((16, 16), 0.5), field_w=3.0, field_h=-1.0)


if __name__ == "__main__":
    unittest.main()
