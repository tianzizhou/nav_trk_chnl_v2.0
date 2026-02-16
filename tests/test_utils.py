# =============================================================================
# Unit tests for src/utils.py
# =============================================================================

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.utils import (
  euler_to_rotation_matrix,
  rotation_matrix_to_euler,
  euler_to_quaternion,
  quaternion_to_euler,
  quaternion_to_rotation_matrix,
  rotation_matrix_to_quaternion,
  rodrigues_to_rotation_matrix,
  camera_to_body,
  body_to_camera,
  body_to_ned,
  ned_to_body,
  normalize_angle,
  normalize_angle_deg,
  unit_vector,
  skew_symmetric,
  load_config,
)


class TestEulerRotationMatrix(unittest.TestCase):
  """Test Euler angle <-> rotation matrix conversions."""

  def test_identity(self):
    """Zero angles should produce identity matrix."""
    R = euler_to_rotation_matrix(0, 0, 0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

  def test_roundtrip_various_angles(self):
    """Euler -> matrix -> Euler should be identity."""
    test_angles = [
      (0.1, 0.2, 0.3),
      (-0.5, 0.3, 1.0),
      (0.0, 0.0, np.pi / 4),
      (np.pi / 6, -np.pi / 3, np.pi / 2),
    ]
    for roll, pitch, yaw in test_angles:
      R = euler_to_rotation_matrix(roll, pitch, yaw)
      r2, p2, y2 = rotation_matrix_to_euler(R)
      R2 = euler_to_rotation_matrix(r2, p2, y2)
      np.testing.assert_allclose(
        R, R2, atol=1e-10,
        err_msg=f"Failed for ({roll}, {pitch}, {yaw})"
      )

  def test_rotation_matrix_orthogonal(self):
    """Rotation matrix should be orthogonal (R^T R = I)."""
    R = euler_to_rotation_matrix(0.5, -0.3, 1.2)
    np.testing.assert_allclose(
      R.T @ R, np.eye(3), atol=1e-12
    )
    self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

  def test_pure_yaw(self):
    """Pure yaw rotation about Z axis."""
    angle = np.pi / 4
    R = euler_to_rotation_matrix(0, 0, angle)
    expected = np.array([
      [np.cos(angle), -np.sin(angle), 0],
      [np.sin(angle), np.cos(angle), 0],
      [0, 0, 1]
    ])
    np.testing.assert_allclose(R, expected, atol=1e-12)


class TestQuaternionConversions(unittest.TestCase):
  """Test quaternion <-> Euler and quaternion <-> matrix."""

  def test_euler_quaternion_roundtrip(self):
    """Euler -> quaternion -> Euler roundtrip."""
    test_angles = [
      (0.1, 0.2, 0.3),
      (-0.4, 0.5, -0.6),
      (0.0, 0.0, 0.0),
    ]
    for roll, pitch, yaw in test_angles:
      q = euler_to_quaternion(roll, pitch, yaw)
      r2, p2, y2 = quaternion_to_euler(q)
      self.assertAlmostEqual(roll, r2, places=10)
      self.assertAlmostEqual(pitch, p2, places=10)
      self.assertAlmostEqual(yaw, y2, places=10)

  def test_quaternion_matrix_roundtrip(self):
    """Quaternion -> matrix -> quaternion roundtrip."""
    q_orig = euler_to_quaternion(0.3, -0.2, 0.8)
    R = quaternion_to_rotation_matrix(q_orig)
    q_back = rotation_matrix_to_quaternion(R)
    # Quaternion sign ambiguity: q and -q represent same rotation
    if q_back[0] * q_orig[0] < 0:
      q_back = -q_back
    np.testing.assert_allclose(q_orig, q_back, atol=1e-10)

  def test_identity_quaternion(self):
    """Identity rotation quaternion is [1, 0, 0, 0]."""
    q = euler_to_quaternion(0, 0, 0)
    np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-12)

  def test_quaternion_unit_norm(self):
    """Quaternion should have unit norm."""
    q = euler_to_quaternion(1.0, -0.5, 2.0)
    self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=10)


class TestRodrigues(unittest.TestCase):
  """Test Rodrigues vector -> rotation matrix."""

  def test_zero_vector(self):
    """Zero vector should give identity."""
    R = rodrigues_to_rotation_matrix([0, 0, 0])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

  def test_90_deg_z(self):
    """90 degree rotation about Z axis."""
    R = rodrigues_to_rotation_matrix([0, 0, np.pi / 2])
    expected = np.array([
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1]
    ], dtype=np.float64)
    np.testing.assert_allclose(R, expected, atol=1e-10)

  def test_orthogonal(self):
    """Result should be orthogonal with det=1."""
    R = rodrigues_to_rotation_matrix([0.3, -0.5, 0.7])
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)


class TestCoordinateTransforms(unittest.TestCase):
  """Test coordinate frame transformations."""

  def test_camera_body_roundtrip(self):
    """Camera->body->camera should recover original point."""
    R = euler_to_rotation_matrix(0.1, 0.2, 0.3)
    T = np.array([0.1, 0.0, -0.05])
    p_cam = np.array([10.0, -5.0, 100.0])

    p_body = camera_to_body(p_cam, R, T)
    p_cam_back = body_to_camera(p_body, R, T)
    np.testing.assert_allclose(p_cam, p_cam_back, atol=1e-10)

  def test_body_ned_roundtrip(self):
    """Body->NED->body should recover original point."""
    roll, pitch, yaw = 0.1, -0.2, 0.5
    p_body = np.array([100.0, 50.0, -30.0])

    p_ned = body_to_ned(p_body, roll, pitch, yaw)
    p_body_back = ned_to_body(p_ned, roll, pitch, yaw)
    np.testing.assert_allclose(p_body, p_body_back, atol=1e-10)

  def test_identity_transform(self):
    """Identity rotation and zero translation should pass through."""
    R = np.eye(3)
    T = np.zeros(3)
    p = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
      camera_to_body(p, R, T), p, atol=1e-12
    )

  def test_full_chain_roundtrip(self):
    """Full chain: camera -> body -> NED -> body -> camera."""
    R_c2b = euler_to_rotation_matrix(0.05, 0.1, 0.0)
    T_c2b = np.array([0.1, 0.0, -0.05])
    roll, pitch, yaw = 0.1, -0.15, 1.0
    p_cam = np.array([5.0, -2.0, 200.0])

    p_body = camera_to_body(p_cam, R_c2b, T_c2b)
    p_ned = body_to_ned(p_body, roll, pitch, yaw)
    p_body_back = ned_to_body(p_ned, roll, pitch, yaw)
    p_cam_back = body_to_camera(p_body_back, R_c2b, T_c2b)

    np.testing.assert_allclose(p_cam, p_cam_back, atol=1e-10)


class TestMathUtils(unittest.TestCase):
  """Test mathematical utility functions."""

  def test_normalize_angle(self):
    """Angle normalization to [-pi, pi]."""
    self.assertAlmostEqual(normalize_angle(0), 0)
    # pi and -pi are both valid boundary values
    self.assertAlmostEqual(
      abs(normalize_angle(np.pi)), np.pi, places=10
    )
    self.assertAlmostEqual(
      abs(normalize_angle(-np.pi)), np.pi, places=10
    )
    # 2*pi wraps to 0
    self.assertAlmostEqual(
      normalize_angle(2 * np.pi), 0.0, places=10
    )
    # General wrapping
    self.assertAlmostEqual(
      normalize_angle(np.pi / 2 + 2 * np.pi),
      np.pi / 2, places=10
    )

  def test_normalize_angle_deg(self):
    """Angle normalization to [-180, 180] degrees."""
    self.assertAlmostEqual(normalize_angle_deg(0), 0)
    self.assertAlmostEqual(normalize_angle_deg(360), 0)
    self.assertAlmostEqual(normalize_angle_deg(-270), 90)

  def test_unit_vector(self):
    """Unit vector should have norm 1."""
    v = unit_vector([3, 4, 0])
    self.assertAlmostEqual(np.linalg.norm(v), 1.0)
    np.testing.assert_allclose(v, [0.6, 0.8, 0.0], atol=1e-12)

  def test_unit_vector_zero(self):
    """Zero vector should return zero vector."""
    v = unit_vector([0, 0, 0])
    np.testing.assert_allclose(v, [0, 0, 0], atol=1e-12)

  def test_skew_symmetric(self):
    """Skew symmetric matrix property: S^T = -S."""
    v = np.array([1.0, 2.0, 3.0])
    S = skew_symmetric(v)
    np.testing.assert_allclose(S, -S.T, atol=1e-12)
    # S * v = 0 (cross product of v with itself)
    np.testing.assert_allclose(S @ v, [0, 0, 0], atol=1e-12)


class TestLoadConfig(unittest.TestCase):
  """Test config loading."""

  def test_load_default_config(self):
    """Should load default config without error."""
    cfg = load_config("config/default_config.yaml")
    self.assertIn('camera', cfg)
    self.assertIn('tracking', cfg)
    self.assertIn('simulation', cfg)
    self.assertEqual(cfg['camera']['image_width'], 1920)
    self.assertEqual(cfg['simulation']['frame_rate'], 120)


if __name__ == '__main__':
  unittest.main()
