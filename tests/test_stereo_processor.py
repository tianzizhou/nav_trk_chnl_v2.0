# =============================================================================
# Unit tests for src/stereo_processor.py
# =============================================================================

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.camera_model import StereoCamera
from src.stereo_processor import StereoProcessor


class TestStereoProcessor(unittest.TestCase):
  """Test stereo processing module."""

  def setUp(self):
    self.cam = StereoCamera(
      config_path="config/default_config.yaml"
    )
    self.proc = StereoProcessor(self.cam)

  def test_depth_from_disparity_known(self):
    """Known disparity should give correct depth."""
    # At 100m: disp = 1200 * 0.3 / 100 = 3.6 px
    depth, conf = self.proc.estimate_depth_direct(
      disparity_value=3.6
    )
    self.assertAlmostEqual(depth, 100.0, places=1)
    self.assertGreater(conf, 0)

  def test_depth_from_size_prior(self):
    """Known bbox height should give correct depth."""
    # At 100m: bbox_h = 0.4 * 1200 / 100 = 4.8 px
    depth, conf = self.proc.estimate_depth_direct(
      bbox_height=4.8
    )
    self.assertAlmostEqual(depth, 100.0, places=1)
    self.assertGreater(conf, 0)

  def test_fusion_near_range(self):
    """Near range: fusion should weight disparity more."""
    proc = StereoProcessor(self.cam)

    # Simulate 50m target
    disp_50m = 1200 * 0.3 / 50  # = 7.2 px
    bbox_h_50m = 0.4 * 1200 / 50  # = 9.6 px

    depth, conf = proc.estimate_depth_direct(
      disparity_value=disp_50m,
      bbox_height=bbox_h_50m
    )
    # Both methods agree, so fusion should be ~50m
    self.assertIsNotNone(depth)
    self.assertAlmostEqual(depth, 50.0, delta=5.0)

  def test_fusion_far_range(self):
    """Far range: fusion should weight size-prior more."""
    proc = StereoProcessor(self.cam)

    # At 1000m, disparity very small and noisy
    # Simulate slight error in disparity
    disp_1000m = 1200 * 0.3 / 1000  # = 0.36 px (very small)
    bbox_h_1000m = 0.4 * 1200 / 1000  # = 0.48 px

    depth, conf = proc.estimate_depth_direct(
      disparity_value=disp_1000m,
      bbox_height=bbox_h_1000m
    )
    self.assertIsNotNone(depth)
    # Should be close to 1000m
    self.assertAlmostEqual(depth, 1000.0, delta=100.0)

  def test_no_input_returns_none(self):
    """No valid inputs should return None."""
    depth, conf = self.proc.estimate_depth_direct()
    self.assertIsNone(depth)
    self.assertEqual(conf, 0.0)

  def test_negative_disparity_ignored(self):
    """Negative disparity should be ignored."""
    depth, conf = self.proc.estimate_depth_direct(
      disparity_value=-1
    )
    self.assertIsNone(depth)

  def test_zero_bbox_height_ignored(self):
    """Zero bbox height should be ignored."""
    depth, conf = self.proc.estimate_depth_direct(
      bbox_height=0
    )
    self.assertIsNone(depth)

  def test_disparity_only_mode(self):
    """Disparity-only mode should ignore size prior."""
    from src.utils import load_config
    config = load_config()
    config['stereo_matching']['depth_method'] = 'disparity'
    proc = StereoProcessor(self.cam, config=config)

    depth, conf = proc.estimate_depth_direct(
      disparity_value=3.6,
      bbox_height=4.8
    )
    self.assertAlmostEqual(depth, 100.0, places=1)

  def test_size_prior_only_mode(self):
    """Size-prior-only mode should ignore disparity."""
    from src.utils import load_config
    config = load_config()
    config['stereo_matching']['depth_method'] = 'size_prior'
    proc = StereoProcessor(self.cam, config=config)

    depth, conf = proc.estimate_depth_direct(
      bbox_height=4.8
    )
    self.assertAlmostEqual(depth, 100.0, places=1)

  def test_depth_precision_at_distances(self):
    """Test depth estimation precision at various distances."""
    distances = [50, 100, 200, 500, 1000]
    for d in distances:
      disp = 1200 * 0.3 / d
      bbox_h = 0.4 * 1200 / d

      depth, conf = self.proc.estimate_depth_direct(
        disparity_value=disp,
        bbox_height=bbox_h
      )
      self.assertIsNotNone(
        depth, msg=f"None depth at {d}m"
      )
      rel_err = abs(depth - d) / d
      self.assertLess(
        rel_err, 0.05,
        msg=f"Error {rel_err:.2%} at {d}m"
      )


if __name__ == '__main__':
  unittest.main()
