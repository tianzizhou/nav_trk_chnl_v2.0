# =============================================================================
# Unit tests for src/position_solver.py
# =============================================================================

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.position_solver import PositionSolver, TargetReport
from src.tracker import TrackState, TrackStatus


class TestPositionSolver(unittest.TestCase):
  """Test position solver."""

  def setUp(self):
    self.solver = PositionSolver()

  def _make_track(self, position, velocity=None, tid=1):
    vel = velocity if velocity is not None else np.zeros(3)
    return TrackState(
      track_id=tid,
      status=TrackStatus.CONFIRMED,
      position=np.array(position, dtype=np.float64),
      velocity=np.array(vel, dtype=np.float64),
      acceleration=np.zeros(3),
      covariance=np.eye(9),
    )

  def test_azimuth_straight_ahead(self):
    """Target straight ahead should have azimuth ~0."""
    ts = self._make_track([0, 0, 500])
    reports = self.solver.solve([ts])
    self.assertEqual(len(reports), 1)
    self.assertAlmostEqual(reports[0].azimuth_deg, 0, places=5)

  def test_azimuth_right(self):
    """Target to the right (positive X) should have positive azimuth."""
    ts = self._make_track([100, 0, 100])  # 45 degrees right
    reports = self.solver.solve([ts])
    self.assertAlmostEqual(
      reports[0].azimuth_deg, 45.0, places=3
    )

  def test_azimuth_left(self):
    """Target to the left (negative X) -> negative azimuth."""
    ts = self._make_track([-100, 0, 100])
    reports = self.solver.solve([ts])
    self.assertAlmostEqual(
      reports[0].azimuth_deg, -45.0, places=3
    )

  def test_elevation_above(self):
    """Target above (negative Y in camera) -> positive elevation."""
    ts = self._make_track([0, -100, 100])
    reports = self.solver.solve([ts])
    self.assertAlmostEqual(
      reports[0].elevation_deg, 45.0, places=3
    )

  def test_elevation_below(self):
    """Target below (positive Y) -> negative elevation."""
    ts = self._make_track([0, 100, 100])
    reports = self.solver.solve([ts])
    self.assertAlmostEqual(
      reports[0].elevation_deg, -45.0, places=3
    )

  def test_slant_range(self):
    """Slant range should be Euclidean distance."""
    ts = self._make_track([300, 400, 0])
    reports = self.solver.solve([ts])
    self.assertAlmostEqual(
      reports[0].slant_range_m, 500.0, places=3
    )

  def test_ttc_approaching(self):
    """Approaching target should have finite TTC."""
    # Target at 1000m, approaching at 100m/s
    ts = self._make_track(
      [0, 0, 1000], velocity=[0, 0, -100]
    )
    reports = self.solver.solve([ts])
    # TTC = 1000 / 100 = 10 seconds
    self.assertAlmostEqual(
      reports[0].ttc_sec, 10.0, places=1
    )

  def test_ttc_receding(self):
    """Receding target should have infinite TTC."""
    ts = self._make_track(
      [0, 0, 1000], velocity=[0, 0, 100]
    )
    reports = self.solver.solve([ts])
    self.assertEqual(reports[0].ttc_sec, float('inf'))

  def test_threat_critical(self):
    """TTC < 2s should be critical."""
    ts = self._make_track(
      [0, 0, 100], velocity=[0, 0, -100]
    )
    reports = self.solver.solve([ts])
    # TTC = 1.0 seconds
    self.assertEqual(reports[0].threat_level, "critical")

  def test_threat_warning(self):
    """2s < TTC < 5s should be warning."""
    ts = self._make_track(
      [0, 0, 300], velocity=[0, 0, -100]
    )
    reports = self.solver.solve([ts])
    # TTC = 3.0 seconds
    self.assertEqual(reports[0].threat_level, "warning")

  def test_threat_safe(self):
    """TTC > 5s should be safe."""
    ts = self._make_track(
      [0, 0, 1000], velocity=[0, 0, -100]
    )
    reports = self.solver.solve([ts])
    # TTC = 10.0 seconds
    self.assertEqual(reports[0].threat_level, "safe")

  def test_multiple_targets(self):
    """Solver should handle multiple targets."""
    tracks = [
      self._make_track([100, 0, 500], tid=1),
      self._make_track([-50, 30, 300], tid=2),
      self._make_track([0, 0, 1000], tid=3),
    ]
    reports = self.solver.solve(tracks)
    self.assertEqual(len(reports), 3)
    ids = [r.track_id for r in reports]
    self.assertEqual(sorted(ids), [1, 2, 3])

  def test_ned_transform_level_flight(self):
    """NED position should be correct for level flight."""
    # Level flight (all angles 0): camera frame ≈ body frame
    # Camera: X=right, Y=down, Z=forward
    # Body: X=forward, Y=right, Z=down
    # With identity cam2body: NED ≈ camera coords
    ts = self._make_track([0, 0, 500])
    reports = self.solver.solve([ts], imu_attitude=[0, 0, 0])
    self.assertIsNotNone(reports[0].position_ned)

  def test_closing_speed_oblique(self):
    """Closing speed for oblique approach should be correct."""
    # Target at [100, 0, 100], distance = sqrt(20000) ≈ 141.4m
    # Velocity toward ego: [-10, 0, -10]
    # Closing speed = -vel . unit_range
    ts = self._make_track(
      [100, 0, 100], velocity=[-10, 0, -10]
    )
    reports = self.solver.solve([ts])
    # unit_range = [100, 0, 100] / 141.4 = [0.707, 0, 0.707]
    # closing = -(-10*0.707 + 0 + -10*0.707) = 14.14
    self.assertAlmostEqual(
      reports[0].closing_speed_mps, 14.14, delta=0.5
    )

  def test_static_methods(self):
    """Test static computation methods."""
    pos = np.array([100, -50, 500])
    vel = np.array([-10, 5, -100])

    az = PositionSolver.compute_azimuth(pos)
    el = PositionSolver.compute_elevation(pos)
    rng = PositionSolver.compute_slant_range(pos)
    ttc = PositionSolver.compute_ttc(pos, vel)

    self.assertIsInstance(az, float)
    self.assertIsInstance(el, float)
    self.assertIsInstance(rng, float)
    self.assertGreater(ttc, 0)


if __name__ == '__main__':
  unittest.main()
