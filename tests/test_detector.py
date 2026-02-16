# =============================================================================
# Unit tests for src/detector.py
# =============================================================================

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.detector import SimulatedDetector, Detection
from src.scene_generator import TargetGT


class TestSimulatedDetector(unittest.TestCase):
  """Test simulated detector."""

  def setUp(self):
    self.det = SimulatedDetector(rng=np.random.RandomState(42))

  def _make_gt(self, distance=100, in_frame=True, visible=True):
    """Helper to create a TargetGT."""
    # bbox size based on distance
    f = 1200.0
    bbox_w = 1.2 * f / max(distance, 1)
    bbox_h = 0.4 * f / max(distance, 1)
    cx, cy = 960, 540
    return TargetGT(
      target_id=1,
      position_cam=np.array([0, 0, distance]),
      velocity_cam=np.array([0, 0, -100]),
      position_ned=np.array([distance, 0, 0]),
      velocity_ned=np.array([-100, 0, 0]),
      bbox_left=np.array([
        cx - bbox_w / 2, cy - bbox_h / 2,
        cx + bbox_w / 2, cy + bbox_h / 2
      ]),
      bbox_right=np.array([
        cx - bbox_w / 2 - 3, cy - bbox_h / 2,
        cx + bbox_w / 2 - 3, cy + bbox_h / 2
      ]),
      in_frame=in_frame,
      distance=distance,
      visible=visible,
    )

  def test_detect_close_target(self):
    """Close target (100m) should always be detected."""
    gt = self._make_gt(distance=100)
    # Run 100 times, should detect most
    detected = 0
    for i in range(100):
      self.det.set_seed(i)
      dets = self.det.detect_from_gt([gt])
      real_dets = [d for d in dets if not d.is_false_alarm]
      if len(real_dets) > 0:
        detected += 1
    # Should detect > 90% at 100m
    self.assertGreater(detected, 90)

  def test_detect_far_target(self):
    """Very far target (3000m) should rarely be detected."""
    gt = self._make_gt(distance=3000)
    detected = 0
    for i in range(100):
      self.det.set_seed(i)
      dets = self.det.detect_from_gt([gt])
      real_dets = [d for d in dets if not d.is_false_alarm]
      if len(real_dets) > 0:
        detected += 1
    # Should detect < 20% at 3000m
    self.assertLess(detected, 20)

  def test_invisible_target_not_detected(self):
    """Invisible (occluded) target should not be detected."""
    gt = self._make_gt(distance=100, visible=False)
    dets = self.det.detect_from_gt([gt])
    real_dets = [d for d in dets if not d.is_false_alarm]
    self.assertEqual(len(real_dets), 0)

  def test_out_of_frame_not_detected(self):
    """Out-of-frame target should not be detected."""
    gt = self._make_gt(distance=100, in_frame=False)
    dets = self.det.detect_from_gt([gt])
    real_dets = [d for d in dets if not d.is_false_alarm]
    self.assertEqual(len(real_dets), 0)

  def test_bbox_noise_added(self):
    """Detection bbox should differ from GT bbox (noise)."""
    gt = self._make_gt(distance=200)
    dets = self.det.detect_from_gt([gt])
    real_dets = [d for d in dets if not d.is_false_alarm]
    if len(real_dets) > 0:
      det = real_dets[0]
      # Bbox should be different from GT due to noise
      diff = np.abs(det.bbox - gt.bbox_left)
      # At 200m, noise is 2.0 * 2.0 = 4.0 std, expect some diff
      self.assertGreater(np.max(diff), 0)

  def test_false_alarms_generated(self):
    """With elevated FA rate, should see false alarms."""
    overrides = {'false_alarm_rate': 5.0}
    gt = self._make_gt(distance=100)
    fa_count = 0
    for i in range(100):
      self.det.set_seed(i)
      dets = self.det.detect_from_gt(
        [gt], detector_overrides=overrides
      )
      fa_count += sum(1 for d in dets if d.is_false_alarm)
    # With rate=5, should average ~5 per frame
    avg_fa = fa_count / 100
    self.assertGreater(avg_fa, 2)

  def test_confidence_decreases_with_distance(self):
    """Confidence should be lower at greater distances."""
    gt_near = self._make_gt(distance=100)
    gt_far = self._make_gt(distance=1000)

    self.det.set_seed(0)
    dets_near = self.det.detect_from_gt([gt_near])
    self.det.set_seed(0)
    dets_far = self.det.detect_from_gt([gt_far])

    near_dets = [d for d in dets_near if not d.is_false_alarm]
    far_dets = [d for d in dets_far if not d.is_false_alarm]

    if near_dets and far_dets:
      self.assertGreater(
        near_dets[0].confidence,
        far_dets[0].confidence
      )

  def test_detection_probability_model(self):
    """Detection probability should decrease with distance."""
    p_100 = self.det._detection_probability(100)
    p_500 = self.det._detection_probability(500)
    p_1500 = self.det._detection_probability(1500)
    p_3000 = self.det._detection_probability(3000)

    self.assertGreater(p_100, 0.9)
    self.assertGreater(p_100, p_500)
    self.assertGreater(p_500, p_1500)
    self.assertGreater(p_1500, p_3000)

  def test_set_seed_reproducible(self):
    """Same seed should produce same results."""
    gt = self._make_gt(distance=200)

    self.det.set_seed(42)
    dets1 = self.det.detect_from_gt([gt])

    self.det.set_seed(42)
    dets2 = self.det.detect_from_gt([gt])

    self.assertEqual(len(dets1), len(dets2))
    for d1, d2 in zip(dets1, dets2):
      np.testing.assert_array_equal(d1.bbox, d2.bbox)
      self.assertEqual(d1.confidence, d2.confidence)


if __name__ == '__main__':
  unittest.main()
