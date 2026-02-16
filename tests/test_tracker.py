# =============================================================================
# Unit tests for src/tracker.py
# =============================================================================

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.tracker import (
  EKFTracker, MultiTargetTracker, TrackStatus,
)


class TestEKFTracker(unittest.TestCase):
  """Test single-target EKF tracker."""

  def test_init_3d(self):
    """Initialize with position only."""
    ekf = EKFTracker([100, 50, 200])
    np.testing.assert_allclose(
      ekf.get_position(), [100, 50, 200], atol=1e-10
    )
    np.testing.assert_allclose(
      ekf.get_velocity(), [0, 0, 0], atol=1e-10
    )

  def test_predict_constant_velocity(self):
    """Prediction with initial velocity should move position."""
    state = [0, 0, 100, 10, 0, 0, 0, 0, 0]
    ekf = EKFTracker(state, dt=0.1)
    ekf.predict()

    pos = ekf.get_position()
    # After 0.1s at 10m/s in X: x = 0 + 10*0.1 = 1.0
    self.assertAlmostEqual(pos[0], 1.0, places=5)

  def test_predict_covariance_grows(self):
    """Covariance should grow after prediction."""
    ekf = EKFTracker([0, 0, 100])
    P_before = np.trace(ekf.P)
    ekf.predict()
    P_after = np.trace(ekf.P)
    self.assertGreater(P_after, P_before)

  def test_update_reduces_covariance(self):
    """Measurement update should reduce covariance."""
    ekf = EKFTracker([0, 0, 100])
    ekf.predict()
    P_before = np.trace(ekf.P)
    ekf.update([0.1, 0.1, 100.5])
    P_after = np.trace(ekf.P)
    self.assertLess(P_after, P_before)

  def test_convergence_constant_position(self):
    """EKF should converge to true position with repeated updates."""
    true_pos = np.array([10, -5, 200])
    ekf = EKFTracker([0, 0, 100], dt=1 / 120)

    for _ in range(50):
      ekf.predict()
      noise = np.random.randn(3) * 1.0  # 1m noise
      ekf.update(true_pos + noise)

    est_pos = ekf.get_position()
    error = np.linalg.norm(est_pos - true_pos)
    self.assertLess(error, 5.0)  # Within 5m after 50 updates

  def test_velocity_estimation(self):
    """EKF should estimate velocity of a moving target."""
    true_vel = np.array([10.0, 0, 0])  # 10 m/s in X
    pos = np.array([0, 0, 200.0])
    ekf = EKFTracker(pos, dt=1 / 120)

    dt = 1 / 120
    for i in range(200):
      pos = pos + true_vel * dt
      ekf.predict()
      noise = np.random.randn(3) * 0.5
      ekf.update(pos + noise)

    est_vel = ekf.get_velocity()
    vel_err = np.linalg.norm(est_vel - true_vel)
    self.assertLess(vel_err, 5.0)  # Within 5 m/s

  def test_predicted_position(self):
    """get_predicted_position should not modify state."""
    ekf = EKFTracker([0, 0, 100, 10, 0, 0, 0, 0, 0])
    state_before = ekf.x.copy()
    pred = ekf.get_predicted_position(0.1)
    np.testing.assert_array_equal(ekf.x, state_before)
    self.assertAlmostEqual(pred[0], 1.0, places=5)


class TestMultiTargetTracker(unittest.TestCase):
  """Test multi-target tracker."""

  def setUp(self):
    self.tracker = MultiTargetTracker()

  def test_create_track_on_first_detection(self):
    """First detection should create a new track."""
    dets = [{'position': [10, -5, 200]}]
    tracks = self.tracker.process_frame(dets)
    self.assertEqual(len(tracks), 1)
    self.assertEqual(tracks[0].status, TrackStatus.TENTATIVE)

  def test_confirm_after_hits(self):
    """Track should be confirmed after N consecutive hits."""
    pos = [10, -5, 200]
    for i in range(5):
      dets = [{'position': pos}]
      tracks = self.tracker.process_frame(dets)

    confirmed = self.tracker.get_confirmed_tracks()
    self.assertGreater(len(confirmed), 0)

  def test_delete_after_misses(self):
    """Track should be deleted after many consecutive misses."""
    # Create track
    dets = [{'position': [10, -5, 200]}]
    self.tracker.process_frame(dets)

    # Miss for many frames
    for _ in range(15):
      self.tracker.process_frame([])

    active = self.tracker.get_active_tracks()
    self.assertEqual(len(active), 0)

  def test_multi_target_association(self):
    """Multiple targets should be tracked independently."""
    # Create two targets
    for _ in range(5):
      dets = [
        {'position': [10, 0, 200]},
        {'position': [-10, 0, 300]},
      ]
      self.tracker.process_frame(dets)

    confirmed = self.tracker.get_confirmed_tracks()
    self.assertEqual(len(confirmed), 2)

    # Verify different track IDs
    ids = [t.track_id for t in confirmed]
    self.assertEqual(len(set(ids)), 2)

  def test_association_correctness(self):
    """Detections should associate with nearest track."""
    # Create two well-separated tracks
    for _ in range(3):
      self.tracker.process_frame([
        {'position': [100, 0, 200]},
        {'position': [-100, 0, 200]},
      ])

    # Feed slightly shifted positions
    self.tracker.process_frame([
      {'position': [101, 1, 201]},
      {'position': [-99, -1, 199]},
    ])

    tracks = self.tracker.get_active_tracks()
    self.assertEqual(len(tracks), 2)

    # No new tracks should have been created
    # (both should match existing)
    all_ids = [t.track_id for t in tracks]
    self.assertEqual(len(set(all_ids)), 2)

  def test_gate_rejects_far_detection(self):
    """Detection far from all tracks should create new track."""
    # Create a track at [100, 0, 200]
    self.tracker.process_frame([
      {'position': [100, 0, 200]}
    ])

    # New detection very far away
    self.tracker.process_frame([
      {'position': [100, 0, 200]},
      {'position': [1000, 0, 500]},
    ])

    # Should now have 2 tracks
    active = self.tracker.get_active_tracks()
    self.assertEqual(len(active), 2)

  def test_reset(self):
    """Reset should clear all tracks."""
    self.tracker.process_frame([
      {'position': [10, -5, 200]}
    ])
    self.tracker.reset()
    self.assertEqual(len(self.tracker.get_active_tracks()), 0)

  def test_predicted_rois(self):
    """Should generate ROIs from predicted positions."""
    # Create and confirm a track
    for _ in range(5):
      self.tracker.process_frame([
        {'position': [0, 0, 200]}
      ])

    rois = self.tracker.get_predicted_rois(
      1920, 1080, 1200, 1200, 960, 540
    )
    self.assertGreater(len(rois), 0)
    # ROI should be a 4-element array
    self.assertEqual(len(rois[0]), 4)


if __name__ == '__main__':
  unittest.main()
