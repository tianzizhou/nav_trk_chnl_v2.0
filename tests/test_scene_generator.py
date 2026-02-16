# =============================================================================
# Unit tests for src/scene_generator.py and scenarios/
# =============================================================================

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.camera_model import StereoCamera
from src.scene_generator import SceneGenerator, FrameData
from scenarios.scenario_base import (
  generate_linear_trajectory,
  generate_accelerating_trajectory,
  generate_sinusoidal_maneuver,
)
from scenarios.s1_head_on import HeadOnScenario
from scenarios.s2_high_speed_cross import HighSpeedCrossScenario
from scenarios.s3_tail_chase import TailChaseScenario
from scenarios.s4_multi_target import MultiTargetScenario
from scenarios.s5_maneuvering import ManeuveringScenario
from scenarios.s6_long_range import LongRangeScenario
from scenarios.s7_clutter import ClutterScenario
from scenarios.s8_occlusion import OcclusionScenario


class TestTrajectoryGenerators(unittest.TestCase):
  """Test trajectory generation utilities."""

  def test_linear_trajectory(self):
    """Linear trajectory should have constant velocity."""
    ts = np.arange(0, 5, 0.1)
    traj = generate_linear_trajectory(
      [0, 0, 0], [10, 0, 0], ts
    )
    # Position at t=1 should be [10, 0, 0]
    idx_1s = 10  # t=1.0s
    np.testing.assert_allclose(
      traj.positions[idx_1s], [10, 0, 0], atol=1e-10
    )
    # All velocities should be [10, 0, 0]
    np.testing.assert_allclose(
      traj.velocities[idx_1s], [10, 0, 0], atol=1e-10
    )

  def test_accelerating_trajectory(self):
    """Accelerating trajectory should follow s = v0*t + 0.5*a*t^2."""
    ts = np.arange(0, 5, 0.01)
    traj = generate_accelerating_trajectory(
      [0, 0, 0], [10, 0, 0], [5, 0, 0], ts
    )
    # At t=2: pos = [10*2 + 0.5*5*4, 0, 0] = [30, 0, 0]
    idx_2s = 200
    np.testing.assert_allclose(
      traj.positions[idx_2s], [30, 0, 0], atol=0.1
    )
    # Velocity at t=2: v = 10 + 5*2 = 20
    np.testing.assert_allclose(
      traj.velocities[idx_2s], [20, 0, 0], atol=0.1
    )

  def test_sinusoidal_maneuver(self):
    """Sinusoidal trajectory should oscillate."""
    ts = np.arange(0, 4, 0.01)
    traj = generate_sinusoidal_maneuver(
      [0, 0, 0], [10, 0, 0],
      amplitude=50, period=2.0,
      maneuver_axis=1, timestamps=ts
    )
    # At t=0, maneuver offset = 0
    self.assertAlmostEqual(traj.positions[0, 1], 0, places=5)
    # At t=0.5 (quarter period), offset = amplitude
    idx = 50  # t=0.5s
    self.assertAlmostEqual(
      traj.positions[idx, 1], 50.0, places=1
    )


class TestScenarios(unittest.TestCase):
  """Test all 8 scenario definitions."""

  def _check_scenario(self, scenario_class, **kwargs):
    """Helper to check a scenario is well-formed."""
    sc = scenario_class(**kwargs)

    self.assertIsInstance(sc.get_name(), str)
    self.assertIsInstance(sc.get_description(), str)

    ego = sc.get_ego_trajectory()
    self.assertEqual(len(ego.positions), sc.num_frames)
    self.assertEqual(len(ego.velocities), sc.num_frames)

    targets = sc.get_target_trajectories()
    self.assertGreater(len(targets), 0)
    for tgt in targets:
      self.assertEqual(len(tgt.positions), sc.num_frames)
      self.assertEqual(len(tgt.velocities), sc.num_frames)

    attitudes = sc.get_ego_attitude()
    self.assertEqual(attitudes.shape, (sc.num_frames, 3))

    return sc

  def test_s1_head_on(self):
    self._check_scenario(HeadOnScenario, duration=5.0)

  def test_s2_high_speed_cross(self):
    self._check_scenario(HighSpeedCrossScenario, duration=5.0)

  def test_s3_tail_chase(self):
    self._check_scenario(TailChaseScenario, duration=5.0)

  def test_s4_multi_target(self):
    sc = self._check_scenario(
      MultiTargetScenario, duration=5.0
    )
    self.assertEqual(len(sc.get_target_trajectories()), 5)

  def test_s5_maneuvering(self):
    self._check_scenario(ManeuveringScenario, duration=5.0)

  def test_s6_long_range(self):
    self._check_scenario(LongRangeScenario, duration=5.0)

  def test_s7_clutter(self):
    self._check_scenario(ClutterScenario, duration=5.0)

  def test_s8_occlusion(self):
    sc = self._check_scenario(OcclusionScenario, duration=5.0)
    occ = sc.get_occlusion_intervals()
    self.assertIn(1, occ)
    self.assertIn(2, occ)


class TestSceneGenerator(unittest.TestCase):
  """Test scene generator frame generation."""

  def setUp(self):
    self.cam = StereoCamera(
      config_path="config/default_config.yaml"
    )
    self.gen = SceneGenerator(self.cam)

  def test_generate_single_frame(self):
    """Generate one frame from S1 scenario."""
    sc = HeadOnScenario(duration=2.0, frame_rate=120)
    frame = self.gen.generate_frame(sc, 0)

    self.assertIsInstance(frame, FrameData)
    self.assertEqual(frame.frame_idx, 0)
    self.assertEqual(
      frame.left_image.shape,
      (1080, 1920, 3)
    )
    self.assertEqual(
      frame.right_image.shape,
      (1080, 1920, 3)
    )
    self.assertEqual(len(frame.ground_truth), 1)

  def test_generate_sequence(self):
    """Generate a short sequence."""
    sc = HeadOnScenario(duration=0.5, frame_rate=120)
    frames = list(self.gen.generate_sequence(
      sc, start_frame=0, end_frame=10
    ))
    self.assertEqual(len(frames), 10)
    # Timestamps should increase
    for i in range(1, len(frames)):
      self.assertGreater(
        frames[i].timestamp,
        frames[i - 1].timestamp
      )

  def test_target_in_frame_head_on(self):
    """S1: target should be in frame at start (2km ahead)."""
    sc = HeadOnScenario(duration=2.0, frame_rate=120)
    frame = self.gen.generate_frame(sc, 0)
    gt = frame.ground_truth[0]
    # Target at 2km should be visible (small but in frame)
    # It's directly ahead so Z_cam should be positive
    self.assertGreater(gt.position_cam[2], 0)

  def test_multi_target_frame(self):
    """S4: should have 5 targets in ground truth."""
    sc = MultiTargetScenario(duration=2.0, frame_rate=120)
    frame = self.gen.generate_frame(sc, 0)
    self.assertEqual(len(frame.ground_truth), 5)

  def test_image_not_black(self):
    """Generated image should not be all black."""
    sc = HeadOnScenario(duration=1.0, frame_rate=120)
    frame = self.gen.generate_frame(sc, 0)
    self.assertGreater(frame.left_image.mean(), 10)
    self.assertGreater(frame.right_image.mean(), 10)

  def test_ground_truth_distance(self):
    """Ground truth distance should be reasonable."""
    sc = HeadOnScenario(duration=2.0, frame_rate=120)
    frame = self.gen.generate_frame(sc, 0)
    gt = frame.ground_truth[0]
    # Initial distance should be roughly 2000m
    self.assertGreater(gt.distance, 1500)
    self.assertLess(gt.distance, 2500)


if __name__ == '__main__':
  unittest.main()
