import unittest

import numpy as np

from anylabeling.views.labeling.view3d.pointcloud import extract_cross_sections


def _make_depth(h=64, w=80):
    """Depth increases with column: depth[u] = u (so it grows left->right)."""
    rows = np.arange(h).astype(np.float32)
    cols = np.arange(w).astype(np.float32)
    return np.broadcast_to(cols[None, :], (h, w)).copy()


class TestExtractCrossSections(unittest.TestCase):
    def test_shapes(self):
        cs = extract_cross_sections(_make_depth(), u=10, v=10, depth_scale=1.0)
        self.assertEqual(cs["left_right_x"].shape, (80,))
        self.assertEqual(cs["left_right_z"].shape, (80,))
        self.assertEqual(cs["top_bottom_y"].shape, (64,))
        self.assertEqual(cs["top_bottom_z"].shape, (64,))

    def test_clamped_beyond_bounds(self):
        cs = extract_cross_sections(_make_depth(), u=1000, v=1000, depth_scale=1.0)
        self.assertEqual(cs["u"], 79.0)
        self.assertEqual(cs["v"], 63.0)
        cs2 = extract_cross_sections(_make_depth(), u=-5, v=-5, depth_scale=1.0)
        self.assertEqual(cs2["u"], 0.0)
        self.assertEqual(cs2["v"], 0.0)

    def test_row_profile_values(self):
        """At row v=10, left->right Z equals depth[10, :]."""
        depth = _make_depth()
        cs = extract_cross_sections(depth, u=0, v=10, depth_scale=2.0)
        expected = depth[10, :] / 2.0
        np.testing.assert_allclose(cs["left_right_z"], expected)

    def test_column_profile_values(self):
        """At column u=10, top->bottom Z equals depth[:, 10] (constant)."""
        depth = _make_depth()
        cs = extract_cross_sections(depth, u=10, v=0, depth_scale=1.0)
        np.testing.assert_allclose(cs["top_bottom_z"], depth[:, 10])

    def test_bounds_match_orthographic_mapping(self):
        """X positions span -(cell_x*(w-1)/2) .. +(cell_x*(w-1)/2)."""
        cs = extract_cross_sections(
            _make_depth(), u=0, v=0,
            depth_scale=1.0, field_width_m=3.0, field_height_m=3.0,
        )
        cell_x = 3.0 / 80
        np.testing.assert_allclose(
            cs["left_right_x"][0], -(80 - 1) / 2.0 * cell_x
        )
        np.testing.assert_allclose(
            cs["left_right_x"][-1], (80 - 1) / 2.0 * cell_x
        )
        cell_y = 3.0 / 64
        np.testing.assert_allclose(
            cs["top_bottom_y"][-1], (64 - 1) / 2.0 * cell_y
        )

    def test_global_range(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        depth[1, 1] = 10.0
        depth[2, 3] = 90.0
        cs = extract_cross_sections(depth, u=1, v=1, depth_scale=1.0)
        # Zero-depth (padding ring) pixels are treated as invalid and excluded,
        # so the usable range spans the real data.
        self.assertAlmostEqual(cs["zmin"], 10.0)
        self.assertAlmostEqual(cs["zmax"], 90.0)


if __name__ == "__main__":
    unittest.main()