# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved.
# Project Name : T2031
# Module Name  : tracker
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Features include:
# 1. Extended Kalman Filter (9-state: pos/vel/acc)
# 2. Multi-target tracker with Hungarian matching
# 3. Track lifecycle management (tentative/confirmed/lost/deleted)
# =============================================================================

import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from src.utils import load_config


class TrackStatus(Enum):
  """Track lifecycle states."""
  TENTATIVE = "tentative"
  CONFIRMED = "confirmed"
  LOST = "lost"
  DELETED = "deleted"


@dataclass
class TrackState:
  """State of a single tracked target."""
  track_id: int
  status: TrackStatus
  position: np.ndarray       # [x, y, z] meters
  velocity: np.ndarray       # [vx, vy, vz] m/s
  acceleration: np.ndarray   # [ax, ay, az] m/s^2
  covariance: np.ndarray     # 9x9 state covariance
  bbox: Optional[np.ndarray] = None  # Last known bbox
  hits: int = 0              # Consecutive hits
  misses: int = 0            # Consecutive misses
  age: int = 0               # Total frames since creation
  last_timestamp: float = 0.0


class EKFTracker:
  """Extended Kalman Filter for single target tracking.

  State vector (9D):
    [x, y, z, vx, vy, vz, ax, ay, az]

  Constant-acceleration motion model with process noise
  modeled on the acceleration component (Singer model).
  """

  def __init__(self, initial_state, initial_cov=None,
               process_noise_accel_std=50.0, dt=1 / 120):
    """Initialize EKF tracker.

    Args:
      initial_state: Initial [x,y,z,vx,vy,vz,ax,ay,az] or
                     [x,y,z] (will pad velocity/accel to 0).
      initial_cov:   Initial 9x9 covariance (or None for default).
      process_noise_accel_std: Process noise on accel (m/s^2).
      dt: Time step (seconds).
    """
    state = np.asarray(initial_state, dtype=np.float64)
    if len(state) == 3:
      state = np.concatenate([state, np.zeros(6)])
    elif len(state) == 6:
      state = np.concatenate([state, np.zeros(3)])

    self.x = state.copy()  # 9D state
    self.dt = dt
    self.accel_std = process_noise_accel_std

    if initial_cov is not None:
      self.P = np.array(initial_cov, dtype=np.float64)
    else:
      self.P = np.diag([
        100, 100, 100,       # position uncertainty (m^2)
        2500, 2500, 2500,    # velocity uncertainty (m/s)^2
        400, 400, 400        # acceleration uncertainty
      ]).astype(np.float64)

    # Measurement matrix: observe position only
    self.H = np.zeros((3, 9))
    self.H[0, 0] = 1
    self.H[1, 1] = 1
    self.H[2, 2] = 1

  def _build_F(self, dt):
    """Build state transition matrix for constant-accel model.

    F = | I  dt*I  0.5*dt^2*I |
        | 0   I      dt*I     |
        | 0   0       I       |
    """
    F = np.eye(9)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    F[0, 6] = 0.5 * dt * dt
    F[1, 7] = 0.5 * dt * dt
    F[2, 8] = 0.5 * dt * dt
    F[3, 6] = dt
    F[4, 7] = dt
    F[5, 8] = dt
    return F

  def _build_Q(self, dt):
    """Build process noise covariance matrix.

    Singer acceleration model: noise enters through acceleration.
    """
    sigma = self.accel_std
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt

    # Process noise for one axis
    q = sigma * sigma
    Q_1d = np.array([
      [dt5 / 20, dt4 / 8, dt3 / 6],
      [dt4 / 8, dt3 / 3, dt2 / 2],
      [dt3 / 6, dt2 / 2, dt],
    ]) * q

    Q = np.zeros((9, 9))
    for i in range(3):
      idx = [i, i + 3, i + 6]
      Q[np.ix_(idx, idx)] = Q_1d

    return Q

  def predict(self, dt=None):
    """Predict state forward by dt seconds.

    Args:
      dt: Time step. If None, use self.dt.
    """
    if dt is None:
      dt = self.dt

    F = self._build_F(dt)
    Q = self._build_Q(dt)

    self.x = F @ self.x
    self.P = F @ self.P @ F.T + Q

  def update(self, measurement, R=None):
    """Update state with a position measurement.

    Args:
      measurement: [x, y, z] observed position (meters).
      R:           3x3 measurement noise covariance.
                   If None, auto-computed from distance.
    """
    z = np.asarray(measurement, dtype=np.float64).flatten()

    if R is None:
      R = self._auto_measurement_noise(z)

    # Kalman gain
    S = self.H @ self.P @ self.H.T + R
    try:
      K = self.P @ self.H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
      return  # Skip update if singular

    # Innovation
    y = z - self.H @ self.x

    self.x = self.x + K @ y
    I_KH = np.eye(9) - K @ self.H
    self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

  def _auto_measurement_noise(self, measurement):
    """Auto-compute measurement noise based on distance.

    Noise scales as Z^2 / (f * B) for stereo depth estimation.

    Args:
      measurement: [x, y, z] position.

    Returns:
      np.ndarray: 3x3 measurement noise covariance.
    """
    z_dist = max(abs(measurement[2]), 1.0)

    # Base noise at 100m = 2m, scales quadratically
    base_sigma = 2.0
    sigma_z = base_sigma * (z_dist / 100.0) ** 2
    sigma_xy = base_sigma * (z_dist / 100.0)

    sigma_z = max(sigma_z, 0.5)
    sigma_xy = max(sigma_xy, 0.5)

    return np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2])

  def get_position(self):
    return self.x[0:3].copy()

  def get_velocity(self):
    return self.x[3:6].copy()

  def get_acceleration(self):
    return self.x[6:9].copy()

  def get_predicted_position(self, dt):
    """Predict position at future time dt without modifying state."""
    F = self._build_F(dt)
    x_pred = F @ self.x
    return x_pred[0:3].copy()


class MultiTargetTracker:
  """Multi-target tracker with data association.

  Manages multiple EKFTracker instances with:
  - Hungarian algorithm for detection-to-track association
  - Track lifecycle management
  - Predicted ROI generation for next frame
  """

  def __init__(self, config=None):
    """Initialize multi-target tracker.

    Args:
      config: Configuration dict (or None for defaults).
    """
    if config is None:
      config = load_config()
    self.config = config

    trk_cfg = config.get('tracking', {})
    ekf_cfg = trk_cfg.get('ekf', {})
    mt_cfg = trk_cfg.get('multi_target', {})

    self.accel_std = ekf_cfg.get('process_noise_accel_std', 50.0)
    self.gate_threshold = mt_cfg.get('association_gate', 50.0)
    self.w_3d = mt_cfg.get('cost_weight_3d_distance', 0.7)
    self.w_iou = mt_cfg.get('cost_weight_iou', 0.3)
    self.hits_to_confirm = mt_cfg.get(
      'tentative_to_confirmed_hits', 3
    )
    self.misses_to_lost = mt_cfg.get(
      'confirmed_to_lost_misses', 5
    )
    self.misses_to_delete = mt_cfg.get(
      'lost_to_deleted_misses', 10
    )
    self.max_tracks = mt_cfg.get('max_tracks', 20)

    self.dt = 1.0 / config.get('simulation', {}).get(
      'frame_rate', 120
    )

    self.tracks = {}       # track_id -> EKFTracker
    self.track_meta = {}   # track_id -> TrackState metadata
    self.next_id = 1
    self.frame_count = 0

  def process_frame(self, detections_3d, timestamp=None):
    """Process one frame of 3D detections.

    Args:
      detections_3d: List of dicts with keys:
        'position': [x, y, z] in camera frame
        'bbox': [x1, y1, x2, y2] (optional)
        'confidence': float (optional)
      timestamp: Current timestamp (optional).

    Returns:
      List[TrackState]: Current track states.
    """
    self.frame_count += 1

    # Step 1: Predict all existing tracks
    for tid, tracker in self.tracks.items():
      tracker.predict(self.dt)

    # Step 2: Associate detections to tracks
    matched, unmatched_dets, unmatched_tracks = (
      self._associate(detections_3d)
    )

    # Step 3: Update matched tracks
    for det_idx, track_id in matched:
      det = detections_3d[det_idx]
      self.tracks[track_id].update(det['position'])

      meta = self.track_meta[track_id]
      meta.hits += 1
      meta.misses = 0
      meta.age += 1
      meta.bbox = det.get('bbox')

      trk = self.tracks[track_id]
      meta.position = trk.get_position()
      meta.velocity = trk.get_velocity()
      meta.acceleration = trk.get_acceleration()
      meta.covariance = trk.P.copy()
      if timestamp is not None:
        meta.last_timestamp = timestamp

      # Status transition: tentative -> confirmed
      if (meta.status == TrackStatus.TENTATIVE
          and meta.hits >= self.hits_to_confirm):
        meta.status = TrackStatus.CONFIRMED

      # Lost -> confirmed (re-acquired)
      if meta.status == TrackStatus.LOST:
        meta.status = TrackStatus.CONFIRMED

    # Step 4: Handle unmatched tracks (missed)
    for track_id in unmatched_tracks:
      meta = self.track_meta[track_id]
      meta.misses += 1
      meta.age += 1

      trk = self.tracks[track_id]
      meta.position = trk.get_position()
      meta.velocity = trk.get_velocity()
      meta.acceleration = trk.get_acceleration()
      meta.covariance = trk.P.copy()

      if (meta.status == TrackStatus.CONFIRMED
          and meta.misses >= self.misses_to_lost):
        meta.status = TrackStatus.LOST

      if meta.misses >= self.misses_to_delete:
        meta.status = TrackStatus.DELETED

    # Step 5: Create new tracks for unmatched detections
    for det_idx in unmatched_dets:
      if len(self.tracks) >= self.max_tracks:
        break

      det = detections_3d[det_idx]
      self._create_track(det, timestamp)

    # Step 6: Remove deleted tracks
    deleted = [
      tid for tid, meta in self.track_meta.items()
      if meta.status == TrackStatus.DELETED
    ]
    for tid in deleted:
      del self.tracks[tid]
      del self.track_meta[tid]

    # Return active track states
    return self.get_active_tracks()

  def get_active_tracks(self):
    """Get all non-deleted track states.

    Returns:
      List[TrackState]: Active tracks.
    """
    return [
      meta for meta in self.track_meta.values()
      if meta.status != TrackStatus.DELETED
    ]

  def get_confirmed_tracks(self):
    """Get only confirmed track states.

    Returns:
      List[TrackState]: Confirmed tracks.
    """
    return [
      meta for meta in self.track_meta.values()
      if meta.status == TrackStatus.CONFIRMED
    ]

  def get_predicted_rois(self, image_width, image_height,
                         fx, fy, cx, cy, margin=100):
    """Get predicted ROIs for next frame.

    Uses EKF predicted positions projected to image plane.

    Args:
      image_width, image_height: Image dimensions.
      fx, fy, cx, cy: Camera intrinsics.
      margin: Extra pixels around predicted position.

    Returns:
      List[np.ndarray]: List of [x1, y1, x2, y2] ROIs.
    """
    rois = []
    for meta in self.track_meta.values():
      if meta.status == TrackStatus.DELETED:
        continue

      tid = meta.track_id
      trk = self.tracks.get(tid)
      if trk is None:
        continue

      pos = trk.get_predicted_position(self.dt)
      if pos[2] <= 0:
        continue

      u = fx * pos[0] / pos[2] + cx
      v = fy * pos[1] / pos[2] + cy

      roi = np.array([
        max(0, u - margin),
        max(0, v - margin),
        min(image_width, u + margin),
        min(image_height, v + margin),
      ])
      rois.append(roi)

    return rois

  def _associate(self, detections_3d):
    """Associate detections to existing tracks.

    Uses Hungarian algorithm on 3D distance cost matrix.

    Returns:
      matched:          List of (det_idx, track_id) tuples.
      unmatched_dets:   List of unmatched detection indices.
      unmatched_tracks: List of unmatched track IDs.
    """
    active_ids = [
      tid for tid, meta in self.track_meta.items()
      if meta.status != TrackStatus.DELETED
    ]

    if not detections_3d or not active_ids:
      return (
        [],
        list(range(len(detections_3d))),
        active_ids,
      )

    # Build cost matrix
    n_det = len(detections_3d)
    n_trk = len(active_ids)
    cost = np.full((n_det, n_trk), 1e6)

    for i, det in enumerate(detections_3d):
      det_pos = np.asarray(det['position'])
      for j, tid in enumerate(active_ids):
        trk_pos = self.tracks[tid].get_position()
        dist_3d = np.linalg.norm(det_pos - trk_pos)

        if dist_3d < self.gate_threshold:
          cost[i, j] = dist_3d

    # Hungarian matching
    row_indices, col_indices = linear_sum_assignment(cost)

    matched = []
    unmatched_dets = set(range(n_det))
    unmatched_tracks = set(active_ids)

    for r, c in zip(row_indices, col_indices):
      if cost[r, c] < self.gate_threshold:
        tid = active_ids[c]
        matched.append((r, tid))
        unmatched_dets.discard(r)
        unmatched_tracks.discard(tid)

    return matched, list(unmatched_dets), list(unmatched_tracks)

  def _create_track(self, detection, timestamp=None):
    """Create a new track from an unmatched detection.

    Args:
      detection: Dict with 'position', optional 'bbox'.
      timestamp: Current timestamp.
    """
    pos = np.asarray(detection['position'], dtype=np.float64)
    tid = self.next_id
    self.next_id += 1

    tracker = EKFTracker(
      initial_state=pos,
      process_noise_accel_std=self.accel_std,
      dt=self.dt,
    )
    self.tracks[tid] = tracker

    self.track_meta[tid] = TrackState(
      track_id=tid,
      status=TrackStatus.TENTATIVE,
      position=pos.copy(),
      velocity=np.zeros(3),
      acceleration=np.zeros(3),
      covariance=tracker.P.copy(),
      bbox=detection.get('bbox'),
      hits=1,
      misses=0,
      age=1,
      last_timestamp=timestamp or 0.0,
    )

  def reset(self):
    """Reset all tracks."""
    self.tracks.clear()
    self.track_meta.clear()
    self.next_id = 1
    self.frame_count = 0
