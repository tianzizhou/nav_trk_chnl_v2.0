# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved.
# Project Name : T2031
# Module Name  : scene_generator
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Features include:
# 1. Synthetic stereo image pair generation
# 2. Ground truth data generation
# 3. NED-to-camera coordinate transformation
# 4. Image noise and motion blur simulation
# =============================================================================

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

from src.camera_model import StereoCamera
from src.utils import (
  euler_to_rotation_matrix,
  load_config,
)


@dataclass
class TargetGT:
  """Ground truth data for one target in one frame."""
  target_id: int
  position_cam: np.ndarray       # [X, Y, Z] camera frame (m)
  velocity_cam: np.ndarray       # [vX, vY, vZ] camera frame (m/s)
  position_ned: np.ndarray       # [N, E, D] NED frame (m)
  velocity_ned: np.ndarray       # [vN, vE, vD] NED frame (m/s)
  bbox_left: Optional[np.ndarray] = None   # [x1,y1,x2,y2] left
  bbox_right: Optional[np.ndarray] = None  # [x1,y1,x2,y2] right
  in_frame: bool = False
  distance: float = 0.0         # Slant range (m)
  visible: bool = True          # Not occluded


@dataclass
class FrameData:
  """Complete data for one simulation frame."""
  timestamp: float
  frame_idx: int
  left_image: np.ndarray         # [H, W, 3] uint8
  right_image: np.ndarray        # [H, W, 3] uint8
  ground_truth: List[TargetGT] = field(default_factory=list)
  imu_attitude: np.ndarray = field(
    default_factory=lambda: np.zeros(3)
  )  # [roll, pitch, yaw]


class SceneGenerator:
  """Generate synthetic stereo image pairs from scenario data.

  Transforms 3D target positions from NED world frame to
  camera frame, generates simple synthetic images, and
  provides ground truth annotations.
  """

  def __init__(self, stereo_camera, config=None):
    """Initialize scene generator.

    Args:
      stereo_camera: StereoCamera instance.
      config:        Configuration dict (or None for defaults).
    """
    self.cam = stereo_camera
    if config is None:
      config = load_config()
    self.config = config

    # Image synthesis parameters
    img_cfg = config.get('simulation', {}).get(
      'image_synthesis', {}
    )
    self.sky_color = np.array(
      img_cfg.get('sky_color', [230, 200, 180]),
      dtype=np.uint8
    )
    self.target_color = np.array(
      img_cfg.get('target_color', [40, 40, 40]),
      dtype=np.uint8
    )
    self.noise_std = img_cfg.get('noise_std', 5.0)
    self.enable_blur = img_cfg.get('enable_motion_blur', True)
    self.blur_length = img_cfg.get('motion_blur_length', 10)

    # Target physical size
    det_cfg = config.get('detection', {})
    tgt_size = det_cfg.get('target_size', {})
    self.target_wingspan = tgt_size.get('wingspan', 1.2)
    self.target_height = tgt_size.get('height', 0.4)

  def generate_frame(self, scenario, frame_idx):
    """Generate one frame of synthetic data.

    Args:
      scenario:  ScenarioBase instance.
      frame_idx: Frame index (0-based).

    Returns:
      FrameData: Complete frame data with images and GT.
    """
    ts = scenario.timestamps[frame_idx]
    ego_traj = scenario.get_ego_trajectory()
    attitudes = scenario.get_ego_attitude()
    targets = scenario.get_target_trajectories()

    # Ego state
    ego_pos = ego_traj.positions[frame_idx]
    ego_att = attitudes[frame_idx]  # [roll, pitch, yaw]

    # Build body-to-NED rotation
    R_body2ned = euler_to_rotation_matrix(
      ego_att[0], ego_att[1], ego_att[2]
    )
    # NED-to-body
    R_ned2body = R_body2ned.T

    # Camera-to-body transform
    R_cam2body = self.cam.R_cam2body
    T_cam2body = self.cam.T_cam2body

    # Combined: NED -> body -> camera
    R_ned2cam = R_cam2body.T @ R_ned2body
    T_ned2cam = R_cam2body.T @ (
      -T_cam2body - R_ned2body @ ego_pos
    )

    # Process each target
    gt_list = []
    for tgt in targets:
      tgt_pos_ned = tgt.positions[frame_idx]
      tgt_vel_ned = tgt.velocities[frame_idx]

      # Relative position in NED
      rel_ned = tgt_pos_ned - ego_pos

      # Transform to camera frame
      pos_cam = R_ned2cam @ tgt_pos_ned + T_ned2cam
      vel_cam = R_ned2cam @ (
        tgt_vel_ned - ego_traj.velocities[frame_idx]
      )

      distance = np.linalg.norm(pos_cam)

      # Check visibility
      gt = TargetGT(
        target_id=tgt.target_id,
        position_cam=pos_cam,
        velocity_cam=vel_cam,
        position_ned=rel_ned,
        velocity_ned=tgt_vel_ned - ego_traj.velocities[
          frame_idx
        ],
        distance=distance,
      )

      # Project to pixels
      if pos_cam[2] > 0:
        uv_l = self.cam.project_to_left(pos_cam)
        uv_r = self.cam.project_to_right(pos_cam)

        if uv_l is not None and uv_r is not None:
          # Compute bounding box from projected size
          bbox_l = self._compute_bbox(
            uv_l, distance, tgt.size
          )
          bbox_r = self._compute_bbox(
            uv_r, distance, tgt.size
          )

          in_left = self.cam.left.is_in_frame(uv_l, margin=-50)
          in_right = self.cam.right.is_in_frame(uv_r, margin=-50)

          if in_left:
            gt.bbox_left = bbox_l
            gt.bbox_right = bbox_r
            gt.in_frame = True

      # Check occlusion (if scenario provides it)
      if hasattr(scenario, 'get_occlusion_intervals'):
        occ = scenario.get_occlusion_intervals()
        if tgt.target_id in occ:
          for t_start, t_end in occ[tgt.target_id]:
            if t_start <= ts <= t_end:
              gt.visible = False

      gt_list.append(gt)

    # Generate synthetic images
    left_img = self._render_image(
      gt_list, is_left=True
    )
    right_img = self._render_image(
      gt_list, is_left=False
    )

    return FrameData(
      timestamp=ts,
      frame_idx=frame_idx,
      left_image=left_img,
      right_image=right_img,
      ground_truth=gt_list,
      imu_attitude=ego_att,
    )

  def generate_sequence(self, scenario, start_frame=0,
                        end_frame=None):
    """Generate a sequence of frames.

    Args:
      scenario:    ScenarioBase instance.
      start_frame: First frame index.
      end_frame:   Last frame index (exclusive). None=all.

    Yields:
      FrameData: One frame at a time.
    """
    if end_frame is None:
      end_frame = scenario.num_frames

    for idx in range(start_frame, end_frame):
      yield self.generate_frame(scenario, idx)

  def _compute_bbox(self, center_px, distance, target_size):
    """Compute bounding box from projected center and distance.

    Args:
      center_px:   [u, v] pixel center.
      distance:    Distance to target (meters).
      target_size: Dict with 'wingspan', 'height'.

    Returns:
      np.ndarray: [x1, y1, x2, y2] bounding box.
    """
    wingspan = target_size.get('wingspan', self.target_wingspan)
    height = target_size.get('height', self.target_height)

    # Projected size in pixels
    if distance < 1.0:
      distance = 1.0
    w_px = wingspan * self.cam.left.fx / distance
    h_px = height * self.cam.left.fy / distance

    # Minimum size: 2x2 pixels
    w_px = max(w_px, 2.0)
    h_px = max(h_px, 2.0)

    x1 = center_px[0] - w_px / 2
    y1 = center_px[1] - h_px / 2
    x2 = center_px[0] + w_px / 2
    y2 = center_px[1] + h_px / 2

    return np.array([x1, y1, x2, y2])

  def _render_image(self, gt_list, is_left=True):
    """Render a synthetic image with sky background and targets.

    Args:
      gt_list:  List of TargetGT for this frame.
      is_left:  True for left camera, False for right.

    Returns:
      np.ndarray: [H, W, 3] uint8 image.
    """
    H = self.cam.left.height
    W = self.cam.left.width

    # Sky background with gradient
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for row in range(H):
      ratio = row / H
      color = (self.sky_color.astype(np.float32)
               * (0.8 + 0.2 * ratio))
      img[row, :] = np.clip(color, 0, 255).astype(np.uint8)

    # Draw targets
    for gt in gt_list:
      if not gt.in_frame:
        continue
      if not gt.visible:
        continue

      bbox = gt.bbox_left if is_left else gt.bbox_right
      if bbox is None:
        continue

      x1 = int(np.clip(bbox[0], 0, W - 1))
      y1 = int(np.clip(bbox[1], 0, H - 1))
      x2 = int(np.clip(bbox[2], 0, W - 1))
      y2 = int(np.clip(bbox[3], 0, H - 1))

      if x2 <= x1 or y2 <= y1:
        continue

      # Draw as a dark ellipse (simple drone shape)
      cx = (x1 + x2) // 2
      cy = (y1 + y2) // 2
      rx = max((x2 - x1) // 2, 1)
      ry = max((y2 - y1) // 2, 1)

      cv2.ellipse(
        img, (cx, cy), (rx, ry), 0, 0, 360,
        self.target_color.tolist(), -1
      )

      # Add a small cross shape for wings
      wing_len = rx
      if wing_len > 2:
        cv2.line(
          img,
          (cx - wing_len, cy),
          (cx + wing_len, cy),
          self.target_color.tolist(), max(1, ry // 2)
        )

    # Add Gaussian noise
    if self.noise_std > 0:
      noise = np.random.randn(H, W, 3) * self.noise_std
      img = np.clip(
        img.astype(np.float32) + noise, 0, 255
      ).astype(np.uint8)

    return img
