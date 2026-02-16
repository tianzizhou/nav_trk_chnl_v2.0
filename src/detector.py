# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved.
# Project Name : T2031
# Module Name  : detector
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Features include:
# 1. Abstract detector interface (for future YOLO integration)
# 2. Simulated detector (GT + configurable noise/miss/FA)
# 3. Distance-dependent detection probability model
# =============================================================================

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.utils import load_config


@dataclass
class Detection:
  """Single detection result."""
  bbox: np.ndarray         # [x1, y1, x2, y2] pixels
  confidence: float        # Detection confidence [0, 1]
  class_id: int = 0        # 0 = drone
  is_false_alarm: bool = False  # For evaluation only


class DetectorBase(ABC):
  """Abstract base class for target detectors.

  All detector implementations must provide a detect() method
  that takes an image and returns a list of detections.
  """

  @abstractmethod
  def detect(self, image, rois=None):
    """Detect targets in image.

    Args:
      image: Input image (np.ndarray [H, W, 3] uint8).
      rois:  Optional list of ROIs to search [[x1,y1,x2,y2],...]
             If None, search the full image.

    Returns:
      List[Detection]: Detected targets.
    """
    pass


class SimulatedDetector(DetectorBase):
  """Simulated detector using ground truth + noise.

  Adds configurable:
  - Bounding box position noise (Gaussian)
  - Distance-dependent detection probability (miss rate)
  - False alarms (Poisson-distributed)
  - Distance-dependent confidence score
  """

  def __init__(self, config=None, rng=None):
    """Initialize simulated detector.

    Args:
      config: Configuration dict. If None, load defaults.
      rng:    numpy RandomState for reproducibility.
    """
    if config is None:
      config = load_config()
    self.config = config

    det_cfg = config.get('detection', {}).get('simulated', {})
    self.bbox_noise_std_base = det_cfg.get(
      'bbox_noise_std_base', 2.0
    )
    self.det_prob_base = det_cfg.get(
      'detection_prob_base', 0.98
    )
    self.det_prob_decay_range = det_cfg.get(
      'detection_prob_decay_range', 1500.0
    )
    self.false_alarm_rate = det_cfg.get(
      'false_alarm_rate', 0.005
    )
    self.conf_threshold = det_cfg.get(
      'confidence_threshold', 0.3
    )

    self.image_width = config['camera']['image_width']
    self.image_height = config['camera']['image_height']

    self.rng = rng or np.random.RandomState()

  def detect(self, image, rois=None):
    """Not used in simulation mode — use detect_from_gt."""
    return []

  def detect_from_gt(self, ground_truth_list, rois=None,
                     detector_overrides=None):
    """Generate detections from ground truth with noise.

    Args:
      ground_truth_list: List of TargetGT objects.
      rois:              Optional ROIs (not used currently).
      detector_overrides: Optional parameter overrides dict.

    Returns:
      List[Detection]: Simulated detections.
    """
    detections = []

    # Override parameters if specified
    fa_rate = self.false_alarm_rate
    if detector_overrides:
      fa_rate = detector_overrides.get(
        'false_alarm_rate', fa_rate
      )

    # Process each ground truth target
    for gt in ground_truth_list:
      if not gt.in_frame or not gt.visible:
        continue

      if gt.bbox_left is None:
        continue

      distance = gt.distance
      if distance <= 0:
        continue

      # Detection probability (Sigmoid decay with distance)
      det_prob = self._detection_probability(distance)
      if self.rng.random() > det_prob:
        continue  # Missed detection

      # Add bbox noise (scaled with distance)
      noise_scale = distance / 100.0
      bbox_noise = (self.rng.randn(4)
                    * self.bbox_noise_std_base
                    * noise_scale)

      noisy_bbox = gt.bbox_left.copy() + bbox_noise

      # Clip to image bounds
      noisy_bbox[0] = np.clip(noisy_bbox[0], 0,
                               self.image_width - 1)
      noisy_bbox[1] = np.clip(noisy_bbox[1], 0,
                               self.image_height - 1)
      noisy_bbox[2] = np.clip(noisy_bbox[2], 0,
                               self.image_width - 1)
      noisy_bbox[3] = np.clip(noisy_bbox[3], 0,
                               self.image_height - 1)

      # Ensure valid bbox
      if (noisy_bbox[2] <= noisy_bbox[0]
          or noisy_bbox[3] <= noisy_bbox[1]):
        continue

      # Confidence (decreases with distance)
      confidence = self._confidence_score(distance)
      if confidence < self.conf_threshold:
        continue

      detections.append(Detection(
        bbox=noisy_bbox,
        confidence=confidence,
        class_id=0,
        is_false_alarm=False,
      ))

    # Generate false alarms
    num_fa = self.rng.poisson(fa_rate)
    for _ in range(num_fa):
      fa_bbox = self._generate_false_alarm()
      detections.append(Detection(
        bbox=fa_bbox,
        confidence=self.rng.uniform(
          self.conf_threshold, 0.5
        ),
        class_id=0,
        is_false_alarm=True,
      ))

    return detections

  def _detection_probability(self, distance):
    """Compute detection probability as function of distance.

    Uses Sigmoid decay model:
    P(d) = base_prob / (1 + exp((d - decay_range) / scale))

    Args:
      distance: Distance to target in meters.

    Returns:
      float: Detection probability [0, 1].
    """
    scale = self.det_prob_decay_range / 5.0
    prob = self.det_prob_base / (
      1.0 + np.exp(
        (distance - self.det_prob_decay_range) / scale
      )
    )
    return float(np.clip(prob, 0, 1))

  def _confidence_score(self, distance):
    """Compute detection confidence as function of distance.

    Args:
      distance: Distance to target in meters.

    Returns:
      float: Confidence score [0, 1].
    """
    # Linear decay with distance + small random variation
    base_conf = np.clip(1.0 - distance / 3000.0, 0.2, 0.99)
    noise = self.rng.randn() * 0.05
    return float(np.clip(base_conf + noise, 0, 1))

  def _generate_false_alarm(self):
    """Generate a random false alarm bounding box.

    Returns:
      np.ndarray: [x1, y1, x2, y2] random bbox.
    """
    # Random position and small size
    w = self.rng.uniform(5, 30)
    h = self.rng.uniform(3, 15)
    x = self.rng.uniform(0, self.image_width - w)
    y = self.rng.uniform(0, self.image_height - h)
    return np.array([x, y, x + w, y + h])

  def set_seed(self, seed):
    """Set random seed for reproducibility."""
    self.rng = np.random.RandomState(seed)
