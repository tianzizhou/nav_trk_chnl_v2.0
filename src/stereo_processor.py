# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved.
# Project Name : T2031
# Module Name  : stereo_processor
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Features include:
# 1. Stereo rectification
# 2. SGBM disparity computation (ROI-based)
# 3. Depth estimation (disparity / size-prior / fusion)
# =============================================================================

import numpy as np
import cv2

from src.camera_model import StereoCamera
from src.utils import load_config


class StereoProcessor:
  """Stereo vision processing: rectification, disparity, depth.

  Provides depth estimation via three methods:
  - Disparity-based (SGBM stereo matching)
  - Size-prior-based (known target physical size)
  - Fusion (weighted combination of both)
  """

  def __init__(self, stereo_camera, config=None):
    """Initialize stereo processor.

    Args:
      stereo_camera: StereoCamera instance.
      config:        Configuration dict (or None for defaults).
    """
    self.cam = stereo_camera
    if config is None:
      config = load_config()
    self.config = config

    sm_cfg = config.get('stereo_matching', {})

    # SGBM parameters
    sgbm_cfg = sm_cfg.get('sgbm', {})
    self.sgbm = cv2.StereoSGBM_create(
      minDisparity=sgbm_cfg.get('min_disparity', 0),
      numDisparities=sgbm_cfg.get('num_disparities', 256),
      blockSize=sgbm_cfg.get('block_size', 5),
      P1=sgbm_cfg.get('p1', 200),
      P2=sgbm_cfg.get('p2', 800),
      disp12MaxDiff=sgbm_cfg.get('disp12_max_diff', 1),
      uniquenessRatio=sgbm_cfg.get('uniqueness_ratio', 10),
      speckleWindowSize=sgbm_cfg.get(
        'speckle_window_size', 100
      ),
      speckleRange=sgbm_cfg.get('speckle_range', 2),
      mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # Depth estimation method
    self.depth_method = sm_cfg.get('depth_method', 'fusion')
    self.fusion_near = sm_cfg.get('fusion_near_range', 100.0)
    self.fusion_far = sm_cfg.get('fusion_far_range', 500.0)

    # Target physical size
    det_cfg = config.get('detection', {})
    tgt_size = det_cfg.get('target_size', {})
    self.target_height = tgt_size.get('height', 0.4)
    self.target_wingspan = tgt_size.get('wingspan', 1.2)

  def compute_disparity_full(self, left_rect, right_rect):
    """Compute full-frame disparity map using SGBM.

    Args:
      left_rect:  Rectified left image (grayscale or BGR).
      right_rect: Rectified right image (grayscale or BGR).

    Returns:
      np.ndarray: Disparity map (float32, pixels).
        Invalid pixels have value <= 0.
    """
    # Convert to grayscale if needed
    if len(left_rect.shape) == 3:
      left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
      right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    else:
      left_gray = left_rect
      right_gray = right_rect

    # SGBM returns disparity * 16 (fixed-point)
    disp16 = self.sgbm.compute(left_gray, right_gray)
    disparity = disp16.astype(np.float32) / 16.0

    # Mark invalid pixels
    disparity[disp16 < 0] = -1.0

    return disparity

  def compute_disparity_roi(self, left_rect, right_rect, bbox,
                            padding=20):
    """Compute disparity for a specific ROI region.

    Only processes the region around the detection bounding box,
    reducing computation significantly.

    Args:
      left_rect:  Rectified left image.
      right_rect: Rectified right image.
      bbox:       [x1, y1, x2, y2] detection bbox in left image.
      padding:    Extra pixels around bbox.

    Returns:
      float: Median disparity in the ROI (pixels), or -1 if
             invalid.
    """
    H, W = left_rect.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(W, int(bbox[2]) + padding)
    y2 = min(H, int(bbox[3]) + padding)

    if x2 <= x1 or y2 <= y1:
      return -1.0

    # Need wider region in right image for disparity search
    num_disp = self.sgbm.getNumDisparities()
    x1_right = max(0, x1 - num_disp)

    # Crop ROI
    left_roi = left_rect[y1:y2, x1:x2]
    right_roi = right_rect[y1:y2, x1_right:x2]

    if left_roi.size == 0 or right_roi.size == 0:
      return -1.0

    # Convert to grayscale
    if len(left_roi.shape) == 3:
      left_roi = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
      right_roi = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    # Ensure minimum size for SGBM
    min_size = self.sgbm.getBlockSize() + num_disp
    if (left_roi.shape[1] < min_size
        or right_roi.shape[1] < min_size):
      return -1.0

    # Create local SGBM with possibly fewer disparities
    local_num_disp = min(
      num_disp,
      ((right_roi.shape[1] - self.sgbm.getBlockSize())
       // 16) * 16
    )
    if local_num_disp < 16:
      return -1.0

    local_sgbm = cv2.StereoSGBM_create(
      minDisparity=0,
      numDisparities=local_num_disp,
      blockSize=self.sgbm.getBlockSize(),
      P1=self.sgbm.getP1(),
      P2=self.sgbm.getP2(),
      disp12MaxDiff=self.sgbm.getDisp12MaxDiff(),
      uniquenessRatio=self.sgbm.getUniquenessRatio(),
      speckleWindowSize=self.sgbm.getSpeckleWindowSize(),
      speckleRange=self.sgbm.getSpeckleRange(),
      mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp16 = local_sgbm.compute(left_roi, right_roi)
    disparity = disp16.astype(np.float32) / 16.0

    # Extract center region (target area)
    offset = x1 - x1_right
    disp_target = disparity[:, offset:offset + (x2 - x1)]

    # Get valid disparities
    valid = disp_target[disp_target > 0]
    if len(valid) == 0:
      return -1.0

    return float(np.median(valid))

  def estimate_depth(self, bbox, left_rect=None,
                     right_rect=None, disparity_value=None):
    """Estimate depth for a detected target.

    Uses the configured depth method (disparity/size/fusion).

    Args:
      bbox:            [x1, y1, x2, y2] detection bbox.
      left_rect:       Rectified left image (for disparity).
      right_rect:      Rectified right image (for disparity).
      disparity_value: Pre-computed disparity (optional).

    Returns:
      tuple: (depth_meters, confidence)
        depth_meters: Estimated depth, or None if failed.
        confidence:   Confidence score [0, 1].
    """
    bbox = np.asarray(bbox, dtype=np.float64)
    bbox_h = bbox[3] - bbox[1]

    # Method 1: Disparity-based depth
    depth_disp = None
    conf_disp = 0.0

    if disparity_value is not None and disparity_value > 0:
      depth_disp = self.cam.depth_from_disparity(
        disparity_value
      )
    elif (left_rect is not None and right_rect is not None):
      disp = self.compute_disparity_roi(
        left_rect, right_rect, bbox
      )
      if disp > 0:
        depth_disp = self.cam.depth_from_disparity(disp)

    if depth_disp is not None and depth_disp > 0:
      # Confidence decreases with distance (disparity gets noisier)
      conf_disp = np.clip(
        1.0 - (depth_disp - 50) / 500, 0.1, 1.0
      )

    # Method 2: Size-prior-based depth
    depth_size = None
    conf_size = 0.0

    if bbox_h > 1:
      depth_size = self.cam.depth_from_target_size(
        bbox_h, self.target_height
      )
      if depth_size is not None and depth_size > 0:
        # Confidence increases with distance (size becomes
        # primary method at range)
        conf_size = np.clip(
          (depth_size - 50) / 500, 0.1, 0.8
        )

    # Fuse or select method
    if self.depth_method == 'disparity':
      return depth_disp, conf_disp

    if self.depth_method == 'size_prior':
      return depth_size, conf_size

    # Fusion mode
    return self._fuse_depth(
      depth_disp, conf_disp, depth_size, conf_size
    )

  def estimate_depth_direct(self, disparity_value=None,
                            bbox_height=None):
    """Simplified depth estimation from disparity and/or bbox.

    Args:
      disparity_value: Disparity in pixels (optional).
      bbox_height:     Bounding box height in pixels (optional).

    Returns:
      tuple: (depth_meters, confidence)
    """
    depth_disp = None
    conf_disp = 0.0

    if disparity_value is not None and disparity_value > 0:
      depth_disp = self.cam.depth_from_disparity(
        disparity_value
      )
      if depth_disp is not None:
        conf_disp = np.clip(
          1.0 - (depth_disp - 50) / 500, 0.1, 1.0
        )

    depth_size = None
    conf_size = 0.0

    if bbox_height is not None and bbox_height > 1:
      depth_size = self.cam.depth_from_target_size(
        bbox_height, self.target_height
      )
      if depth_size is not None:
        conf_size = np.clip(
          (depth_size - 50) / 500, 0.1, 0.8
        )

    if self.depth_method == 'disparity':
      return depth_disp, conf_disp
    if self.depth_method == 'size_prior':
      return depth_size, conf_size

    return self._fuse_depth(
      depth_disp, conf_disp, depth_size, conf_size
    )

  def _fuse_depth(self, depth_disp, conf_disp,
                  depth_size, conf_size):
    """Fuse disparity and size-prior depth estimates.

    Near range: weight toward disparity.
    Far range: weight toward size-prior.

    Args:
      depth_disp, conf_disp: Disparity-based estimate.
      depth_size, conf_size: Size-prior-based estimate.

    Returns:
      tuple: (fused_depth, fused_confidence)
    """
    if depth_disp is None and depth_size is None:
      return None, 0.0

    if depth_disp is None:
      return depth_size, conf_size

    if depth_size is None:
      return depth_disp, conf_disp

    # Distance-dependent fusion weight
    # Use average of both estimates as reference
    avg_depth = (depth_disp + depth_size) / 2

    if avg_depth <= self.fusion_near:
      w_disp = 1.0
    elif avg_depth >= self.fusion_far:
      w_disp = 0.0
    else:
      # Linear interpolation
      w_disp = (
        (self.fusion_far - avg_depth)
        / (self.fusion_far - self.fusion_near)
      )

    w_size = 1.0 - w_disp

    fused_depth = w_disp * depth_disp + w_size * depth_size
    fused_conf = w_disp * conf_disp + w_size * conf_size

    return fused_depth, fused_conf
