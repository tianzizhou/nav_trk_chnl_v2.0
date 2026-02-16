# =============================================================================
# Integration tests for src/pipeline.py
# =============================================================================

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import DetectionPipeline, FrameResult
from scenarios.s1_head_on import HeadOnScenario
from scenarios.s4_multi_target import MultiTargetScenario


class TestPipeline(unittest.TestCase):
  """Integration tests for the full detection pipeline."""

  def setUp(self):
    self.pipeline = DetectionPipeline()

  def test_single_frame_processing(self):
    """Process a single frame without errors."""
    scenario = HeadOnScenario(duration=0.1)
    frame_data = self.pipeline.scene_gen.generate_frame(
      scenario, 0
    )
    result = self.pipeline.process_frame(frame_data)

    self.assertIsInstance(result, FrameResult)
    self.assertEqual(result.frame_idx, 0)
    self.assertIn('total_ms', result.timing_ms)
    self.assertGreater(result.timing_ms['total_ms'], 0)

  def test_sequence_s1(self):
    """Run S1 head-on for a short duration."""
    scenario = HeadOnScenario(duration=0.5, frame_rate=30)
    result = self.pipeline.run_sequence(scenario)

    self.assertEqual(result.scenario_name, "S1_Head_On")
    self.assertEqual(result.num_frames, 15)
    self.assertGreater(result.total_time_sec, 0)
    self.assertIn('detection_rate', result.metrics)
    self.assertIn('avg_frame_time_ms', result.metrics)

  def test_sequence_produces_detections(self):
    """Pipeline should produce some detections."""
    # Use a closer-range scenario for reliable detection
    scenario = HeadOnScenario(duration=2.0, frame_rate=10)
    result = self.pipeline.run_sequence(scenario)

    frames_with_dets = sum(
      1 for fr in result.frame_results
      if fr.num_detections > 0
    )
    self.assertGreater(frames_with_dets, 0)

  def test_sequence_produces_tracks(self):
    """Pipeline should produce confirmed tracks."""
    scenario = HeadOnScenario(duration=10.0, frame_rate=10)
    result = self.pipeline.run_sequence(scenario)

    frames_with_tracks = sum(
      1 for fr in result.frame_results
      if fr.num_confirmed_tracks > 0
    )
    self.assertGreater(frames_with_tracks, 0)

  def test_sequence_produces_reports(self):
    """Pipeline should produce target reports."""
    scenario = HeadOnScenario(duration=10.0, frame_rate=10)
    result = self.pipeline.run_sequence(scenario)

    frames_with_reports = sum(
      1 for fr in result.frame_results
      if len(fr.target_reports) > 0
    )
    self.assertGreater(frames_with_reports, 0)

  def test_target_reports_have_valid_fields(self):
    """Target reports should have valid azimuth/range/TTC."""
    scenario = HeadOnScenario(duration=2.0, frame_rate=10)
    result = self.pipeline.run_sequence(scenario)

    for fr in result.frame_results:
      for report in fr.target_reports:
        self.assertIsNotNone(report.azimuth_deg)
        self.assertIsNotNone(report.slant_range_m)
        self.assertGreater(report.slant_range_m, 0)
        self.assertIn(
          report.threat_level,
          ["safe", "warning", "critical"]
        )

  def test_multi_target_scenario(self):
    """S4 multi-target should track multiple targets."""
    scenario = MultiTargetScenario(duration=2.0, frame_rate=10)
    result = self.pipeline.run_sequence(scenario)
    self.assertEqual(result.num_frames, 20)

    max_tracks = max(
      fr.num_confirmed_tracks
      for fr in result.frame_results
    )
    self.assertGreater(max_tracks, 0)

  def test_metrics_computed(self):
    """Metrics should be computed for a sequence."""
    scenario = HeadOnScenario(duration=0.5, frame_rate=30)
    result = self.pipeline.run_sequence(scenario)

    m = result.metrics
    self.assertIn('detection_rate', m)
    self.assertIn('false_alarm_rate', m)
    self.assertIn('avg_frame_time_ms', m)
    self.assertIn('max_frame_time_ms', m)

    self.assertGreaterEqual(m['detection_rate'], 0)
    self.assertLessEqual(m['detection_rate'], 1)

  def test_iou_computation(self):
    """Test IoU helper function."""
    # Perfect overlap
    iou = DetectionPipeline._compute_iou(
      [0, 0, 10, 10], [0, 0, 10, 10]
    )
    self.assertAlmostEqual(iou, 1.0, places=5)

    # No overlap
    iou = DetectionPipeline._compute_iou(
      [0, 0, 5, 5], [10, 10, 15, 15]
    )
    self.assertAlmostEqual(iou, 0.0, places=5)

    # Partial overlap
    iou = DetectionPipeline._compute_iou(
      [0, 0, 10, 10], [5, 5, 15, 15]
    )
    # Intersection = 5*5=25, union = 100+100-25=175
    self.assertAlmostEqual(iou, 25 / 175, places=5)

  def test_timing_breakdown(self):
    """Timing should be broken down by module."""
    scenario = HeadOnScenario(duration=0.2, frame_rate=30)
    result = self.pipeline.run_sequence(scenario)

    for fr in result.frame_results:
      self.assertIn('detection_ms', fr.timing_ms)
      self.assertIn('depth_ms', fr.timing_ms)
      self.assertIn('tracking_ms', fr.timing_ms)
      self.assertIn('solving_ms', fr.timing_ms)
      self.assertIn('total_ms', fr.timing_ms)


if __name__ == '__main__':
  unittest.main()
