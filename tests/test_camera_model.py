# =============================================================================
# Unit tests for src/camera_model.py
# =============================================================================

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.camera_model import MonoCamera, StereoCamera


class TestMonoCamera(unittest.TestCase):
  """Test monocular camera model."""

  def setUp(self):
    """Create a standard camera for testing."""
    self.cam = MonoCamera(
      fx=1200, fy=1200, cx=960, cy=540,
      dist=[0, 0, 0, 0, 0],  # No distortion for basic tests
      width=1920, height=1080
    )

  def test_project_center(self):
    """Point on optical axis should project to principal point."""
    p = np.array([0, 0, 100])
    uv = self.cam.project(p)
    np.testing.assert_allclose(uv, [960, 540], atol=1e-10)

  def test_project_behind_camera(self):
    """Point behind camera should return None."""
    p = np.array([0, 0, -10])
    self.assertIsNone(self.cam.project(p))

  def test_project_backproject_roundtrip(self):
    """project -> back_project should recover 3D point."""
    test_points = [
      [0, 0, 100],
      [10, -5, 200],
      [-20, 15, 50],
      [50, 30, 500],
      [1, 1, 2000],
    ]
    for p in test_points:
      p = np.array(p, dtype=np.float64)
      uv = self.cam.project(p)
      self.assertIsNotNone(uv)
      p_back = self.cam.back_project(uv, p[2])
      np.testing.assert_allclose(
        p, p_back, atol=1e-10,
        err_msg=f"Failed for point {p}"
      )

  def test_project_backproject_with_distortion(self):
    """Roundtrip with distortion applied and removed."""
    cam_dist = MonoCamera(
      fx=1200, fy=1200, cx=960, cy=540,
      dist=[-0.1, 0.01, 0.001, -0.001, 0.0],
      width=1920, height=1080
    )
    test_points = [
      [0, 0, 100],
      [5, -3, 200],
      [-10, 8, 150],
    ]
    for p in test_points:
      p = np.array(p, dtype=np.float64)
      uv = cam_dist.project(p, apply_distortion=True)
      self.assertIsNotNone(uv)
      p_back = cam_dist.back_project(
        uv, p[2], remove_distortion=True
      )
      np.testing.assert_allclose(
        p, p_back, atol=1e-4,
        err_msg=f"Distortion roundtrip failed for {p}"
      )

  def test_is_in_frame(self):
    """Test frame boundary check."""
    self.assertTrue(self.cam.is_in_frame([960, 540]))
    self.assertTrue(self.cam.is_in_frame([0, 0]))
    self.assertFalse(self.cam.is_in_frame([-1, 540]))
    self.assertFalse(self.cam.is_in_frame([1920, 540]))
    self.assertFalse(self.cam.is_in_frame([960, 1080]))
    self.assertFalse(self.cam.is_in_frame(None))

  def test_pixel_to_ray_center(self):
    """Ray through center should be [0, 0, 1]."""
    ray = self.cam.pixel_to_ray([960, 540])
    np.testing.assert_allclose(ray, [0, 0, 1], atol=1e-10)

  def test_pixel_to_ray_unit_norm(self):
    """Ray should be unit vector."""
    ray = self.cam.pixel_to_ray([100, 200])
    self.assertAlmostEqual(np.linalg.norm(ray), 1.0, places=10)

  def test_various_depths(self):
    """Test at different distances: 50, 200, 500, 1000, 2000m."""
    depths = [50, 200, 500, 1000, 2000]
    p_base = np.array([5.0, -3.0, 1.0])  # Direction
    for d in depths:
      p = p_base * d
      uv = self.cam.project(p)
      self.assertIsNotNone(uv)
      p_back = self.cam.back_project(uv, p[2])
      np.testing.assert_allclose(
        p, p_back, atol=1e-8,
        err_msg=f"Failed at depth {d}m"
      )


class TestStereoCamera(unittest.TestCase):
  """Test stereo camera model."""

  def setUp(self):
    """Create stereo camera from default config."""
    self.stereo = StereoCamera(
      config_path="config/default_config.yaml"
    )

  def test_baseline(self):
    """Baseline should match config."""
    self.assertAlmostEqual(self.stereo.baseline, 0.3)

  def test_left_right_projection_disparity(self):
    """Left and right projections should differ by disparity."""
    p = np.array([0, 0, 100])  # On optical axis, 100m away

    uv_l = self.stereo.project_to_left(p)
    uv_r = self.stereo.project_to_right(p)

    self.assertIsNotNone(uv_l)
    self.assertIsNotNone(uv_r)

    # Expected disparity = f * B / Z = 1200 * 0.3 / 100 = 3.6
    expected_disp = 1200.0 * 0.3 / 100.0
    actual_disp = uv_l[0] - uv_r[0]
    self.assertAlmostEqual(actual_disp, expected_disp, places=5)

    # Vertical coordinates should be same (rectified)
    self.assertAlmostEqual(uv_l[1], uv_r[1], places=5)

  def test_depth_from_disparity(self):
    """Depth calculation: Z = f * B / d."""
    # At 100m, disparity = 1200 * 0.3 / 100 = 3.6 pixels
    depth = self.stereo.depth_from_disparity(3.6)
    self.assertAlmostEqual(depth, 100.0, places=5)

    # At 50m, disparity = 7.2
    depth = self.stereo.depth_from_disparity(7.2)
    self.assertAlmostEqual(depth, 50.0, places=5)

  def test_depth_from_disparity_zero(self):
    """Zero disparity should return None."""
    self.assertIsNone(self.stereo.depth_from_disparity(0))
    self.assertIsNone(self.stereo.depth_from_disparity(-1))

  def test_disparity_from_depth(self):
    """Disparity from depth: d = f * B / Z."""
    disp = self.stereo.disparity_from_depth(100)
    self.assertAlmostEqual(disp, 3.6, places=5)

  def test_depth_disparity_roundtrip(self):
    """depth -> disparity -> depth roundtrip."""
    depths = [50, 100, 200, 500, 1000]
    for d in depths:
      disp = self.stereo.disparity_from_depth(d)
      d_back = self.stereo.depth_from_disparity(disp)
      self.assertAlmostEqual(
        d, d_back, places=8,
        msg=f"Roundtrip failed at depth {d}m"
      )

  def test_depth_from_target_size(self):
    """Depth from known target size."""
    # Target 0.4m tall, appearing as 48 pixels
    # Z = 0.4 * 1200 / 48 = 10m
    depth = self.stereo.depth_from_target_size(48, 0.4)
    self.assertAlmostEqual(depth, 10.0, places=5)

  def test_triangulate_known_point(self):
    """Triangulate a known 3D point from stereo pixels."""
    p_true = np.array([5.0, -3.0, 100.0])

    uv_l = self.stereo.project_to_left(p_true)
    uv_r = self.stereo.project_to_right(p_true)

    p_est = self.stereo.triangulate(uv_l, uv_r)
    np.testing.assert_allclose(
      p_est, p_true, atol=0.1,
      err_msg="Triangulation error too large"
    )

  def test_triangulate_various_distances(self):
    """Triangulate at various distances."""
    test_points = [
      [2, 1, 50],
      [10, -5, 100],
      [20, 10, 200],
      [50, -20, 500],
    ]
    for pt in test_points:
      p_true = np.array(pt, dtype=np.float64)
      uv_l = self.stereo.project_to_left(p_true)
      uv_r = self.stereo.project_to_right(p_true)

      if uv_l is not None and uv_r is not None:
        p_est = self.stereo.triangulate(uv_l, uv_r)
        rel_err = np.linalg.norm(p_est - p_true) / p_true[2]
        self.assertLess(
          rel_err, 0.01,
          msg=f"Triangulation error {rel_err:.4f} at {pt}"
        )

  def test_pixel_to_3d(self):
    """pixel_to_3d should recover the 3D point."""
    p_true = np.array([10, -5, 200])
    uv = self.stereo.project_to_left(p_true)
    p_est = self.stereo.pixel_to_3d(uv, p_true[2])
    np.testing.assert_allclose(p_est, p_true, atol=1e-8)


if __name__ == '__main__':
  unittest.main()
