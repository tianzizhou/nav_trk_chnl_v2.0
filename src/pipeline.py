# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved.
# Project Name : T2031
# Module Name  : pipeline
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Features include:
# 1. Full single-frame processing pipeline
# 2. Sequence-level processing with state persistence
# 3. Per-module timing profiling
# 4. Result recording and evaluation
# =============================================================================

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from src.camera_model import StereoCamera
from src.scene_generator import SceneGenerator
from src.stereo_processor import StereoProcessor
from src.detector import SimulatedDetector
from src.tracker import MultiTargetTracker
from src.position_solver import PositionSolver, TargetReport
from src.utils import load_config


@dataclass
class FrameResult:
  """Result of processing one frame."""
  frame_idx: int
  timestamp: float
  detections: list = field(default_factory=list)
  tracks: list = field(default_factory=list)
  target_reports: List[TargetReport] = field(
    default_factory=list
  )
  ground_truth: list = field(default_factory=list)
  timing_ms: Dict[str, float] = field(default_factory=dict)
  num_detections: int = 0
  num_confirmed_tracks: int = 0


@dataclass
class SequenceResult:
  """Result of processing a full scenario sequence."""
  scenario_name: str
  num_frames: int
  frame_results: List[FrameResult] = field(
    default_factory=list
  )
  total_time_sec: float = 0.0
  avg_frame_time_ms: float = 0.0
  metrics: Dict[str, float] = field(default_factory=dict)


class DetectionPipeline:
  """Complete stereo vision UAV detection pipeline.

  Integrates all modules:
  SceneGenerator → Detector → StereoProcessor →
  Tracker → PositionSolver
  """

  def __init__(self, config=None, config_path=None):
    """Initialize the pipeline.

    Args:
      config:      Configuration dict.
      config_path: Path to config file (used if config is None).
    """
    if config is None:
      config_path = config_path or "config/default_config.yaml"
      config = load_config(config_path)
    self.config = config

    # Initialize all modules
    self.camera = StereoCamera(config=config)
    self.scene_gen = SceneGenerator(self.camera, config)
    self.stereo = StereoProcessor(self.camera, config)
    self.detector = SimulatedDetector(config=config)
    self.tracker = MultiTargetTracker(config=config)
    self.solver = PositionSolver(self.camera, config)

    self._detector_overrides = None

    # Detection schedule
    self.full_detect_interval = config.get(
      'detection', {}
    ).get('full_detect_interval', 4)

    # Seed
    seed = config.get('simulation', {}).get('random_seed', 42)
    self.detector.set_seed(seed)

  def process_frame(self, frame_data, frame_count=0):
    """Process a single frame through the full pipeline.

    Args:
      frame_data: FrameData from SceneGenerator.
      frame_count: Current frame count (for full-detect schedule).

    Returns:
      FrameResult: Processing result.
    """
    timing = {}
    t0 = time.perf_counter()

    # Step 1: Detection (simulated from GT)
    t1 = time.perf_counter()

    gt_list = frame_data.ground_truth

    # Pass scenario-specific detector overrides
    detections = self.detector.detect_from_gt(
      gt_list,
      detector_overrides=self._detector_overrides,
    )
    timing['detection_ms'] = (
      (time.perf_counter() - t1) * 1000
    )

    # Step 2: Depth estimation for each detection
    t2 = time.perf_counter()
    detections_3d = []
    for det in detections:
      bbox = det.bbox
      bbox_h = bbox[3] - bbox[1]

      # Use GT disparity for simulation (since synthetic images
      # may not have good texture for SGBM)
      # Find nearest GT to this detection
      disparity = None
      for gt in gt_list:
        if gt.bbox_left is not None and gt.in_frame:
          iou = self._compute_iou(bbox, gt.bbox_left)
          if iou > 0.3:
            disparity = self.camera.disparity_from_depth(
              gt.distance
            )
            break

      depth, conf = self.stereo.estimate_depth_direct(
        disparity_value=disparity,
        bbox_height=bbox_h if bbox_h > 0 else None,
      )

      if depth is not None and depth > 0:
        center_u = (bbox[0] + bbox[2]) / 2
        center_v = (bbox[1] + bbox[3]) / 2
        pos_3d = self.camera.pixel_to_3d(
          [center_u, center_v], depth
        )
        detections_3d.append({
          'position': pos_3d,
          'bbox': bbox,
          'confidence': det.confidence,
          'depth': depth,
          'is_false_alarm': det.is_false_alarm,
        })

    timing['depth_ms'] = (
      (time.perf_counter() - t2) * 1000
    )

    # Step 3: Multi-target tracking
    t3 = time.perf_counter()
    track_states = self.tracker.process_frame(
      detections_3d, timestamp=frame_data.timestamp
    )
    timing['tracking_ms'] = (
      (time.perf_counter() - t3) * 1000
    )

    # Step 4: Position solving
    t4 = time.perf_counter()
    confirmed_tracks = self.tracker.get_confirmed_tracks()
    target_reports = self.solver.solve(
      confirmed_tracks,
      imu_attitude=frame_data.imu_attitude,
    )
    timing['solving_ms'] = (
      (time.perf_counter() - t4) * 1000
    )

    timing['total_ms'] = (
      (time.perf_counter() - t0) * 1000
    )

    return FrameResult(
      frame_idx=frame_data.frame_idx,
      timestamp=frame_data.timestamp,
      detections=detections,
      tracks=track_states,
      target_reports=target_reports,
      ground_truth=gt_list,
      timing_ms=timing,
      num_detections=len(detections),
      num_confirmed_tracks=len(confirmed_tracks),
    )

  def run_sequence(self, scenario, verbose=False):
    """Run the pipeline on a full scenario sequence.

    Args:
      scenario: ScenarioBase instance.
      verbose:  Print progress every 100 frames.

    Returns:
      SequenceResult: Complete sequence results.
    """
    self.tracker.reset()
    seed = self.config.get('simulation', {}).get(
      'random_seed', 42
    )
    self.detector.set_seed(seed)

    # Capture scenario-specific detector overrides
    self._detector_overrides = None
    if hasattr(scenario, 'get_detector_overrides'):
      self._detector_overrides = scenario.get_detector_overrides()

    frame_results = []
    t_start = time.perf_counter()

    for frame_data in self.scene_gen.generate_sequence(scenario):
      result = self.process_frame(
        frame_data, frame_count=frame_data.frame_idx
      )
      frame_results.append(result)

      if verbose and frame_data.frame_idx % 100 == 0:
        print(
          f"  Frame {frame_data.frame_idx}/"
          f"{scenario.num_frames}: "
          f"dets={result.num_detections}, "
          f"tracks={result.num_confirmed_tracks}, "
          f"time={result.timing_ms.get('total_ms', 0):.1f}ms"
        )

    total_time = time.perf_counter() - t_start

    seq_result = SequenceResult(
      scenario_name=scenario.get_name(),
      num_frames=len(frame_results),
      frame_results=frame_results,
      total_time_sec=total_time,
      avg_frame_time_ms=(
        total_time * 1000 / max(len(frame_results), 1)
      ),
    )

    # Compute evaluation metrics
    seq_result.metrics = self._compute_metrics(
      frame_results
    )

    return seq_result

  def _compute_metrics(self, frame_results):
    """Compute evaluation metrics over a sequence.

    Args:
      frame_results: List of FrameResult.

    Returns:
      dict: Evaluation metrics.
    """
    metrics = {}

    total_gt_visible = 0
    total_detected = 0
    total_false_alarms = 0
    total_frames = len(frame_results)

    position_errors = []
    azimuth_errors = []
    range_errors = []
    ttc_errors = []

    for fr in frame_results:
      # Count visible GT targets
      gt_visible = [
        g for g in fr.ground_truth
        if g.in_frame and g.visible
      ]
      total_gt_visible += len(gt_visible)

      # Count real detections (not false alarms)
      real_dets = [
        d for d in fr.detections if not d.is_false_alarm
      ]
      total_detected += len(real_dets)

      # Count false alarms
      fa = [d for d in fr.detections if d.is_false_alarm]
      total_false_alarms += len(fa)

      # Position accuracy (match reports to GT)
      for report in fr.target_reports:
        # Find closest GT
        best_dist = float('inf')
        best_gt = None
        for gt in gt_visible:
          dist = np.linalg.norm(
            report.position_cam - gt.position_cam
          )
          if dist < best_dist:
            best_dist = dist
            best_gt = gt

        if best_gt is not None and best_gt.distance > 0:
          # 3D position error
          pos_err = np.linalg.norm(
            report.position_cam - best_gt.position_cam
          )
          position_errors.append(pos_err)

          # Relative range error
          range_err = abs(
            report.slant_range_m - best_gt.distance
          ) / best_gt.distance
          range_errors.append(range_err)

          # Azimuth error
          gt_az = np.degrees(np.arctan2(
            best_gt.position_cam[0],
            best_gt.position_cam[2]
          ))
          az_err = abs(report.azimuth_deg - gt_az)
          azimuth_errors.append(az_err)

    # Detection rate
    metrics['detection_rate'] = (
      total_detected / max(total_gt_visible, 1)
    )

    # False alarm rate
    metrics['false_alarm_rate'] = (
      total_false_alarms / max(total_frames, 1)
    )

    # Position RMSE
    if position_errors:
      metrics['position_rmse_m'] = float(np.sqrt(
        np.mean(np.array(position_errors) ** 2)
      ))
    else:
      metrics['position_rmse_m'] = float('nan')

    # Range relative error
    if range_errors:
      metrics['range_rel_error_mean'] = float(
        np.mean(range_errors)
      )
    else:
      metrics['range_rel_error_mean'] = float('nan')

    # Azimuth error
    if azimuth_errors:
      metrics['azimuth_error_deg_mean'] = float(
        np.mean(azimuth_errors)
      )
    else:
      metrics['azimuth_error_deg_mean'] = float('nan')

    # Timing stats
    total_times = [
      fr.timing_ms.get('total_ms', 0)
      for fr in frame_results
    ]
    if total_times:
      metrics['avg_frame_time_ms'] = float(
        np.mean(total_times)
      )
      metrics['max_frame_time_ms'] = float(
        np.max(total_times)
      )

    return metrics

  @staticmethod
  def _compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes.

    Args:
      bbox1, bbox2: [x1, y1, x2, y2] arrays.

    Returns:
      float: IoU value [0, 1].
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter

    if union <= 0:
      return 0.0
    return inter / union
