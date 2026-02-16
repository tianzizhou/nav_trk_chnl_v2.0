# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved.
# Project Name : T2031
# Module Name  : position_solver
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Features include:
# 1. Azimuth / elevation angle computation
# 2. Slant range computation
# 3. Relative velocity estimation
# 4. Time-to-collision (TTC) computation
# 5. Camera-to-NED coordinate transformation
# =============================================================================

import numpy as np
from dataclasses import dataclass
from typing import List

from src.utils import (
  euler_to_rotation_matrix,
  camera_to_body,
  body_to_ned,
  load_config,
)


@dataclass
class TargetReport:
  """Resolved target report with position and threat info."""
  track_id: int
  azimuth_deg: float        # Azimuth angle (degrees)
  elevation_deg: float      # Elevation angle (degrees)
  slant_range_m: float      # Slant range (meters)
  velocity_mps: float       # Relative speed (m/s)
  closing_speed_mps: float  # Radial closing speed (m/s)
  ttc_sec: float            # Time-to-collision (seconds)
  position_cam: np.ndarray  # [X, Y, Z] camera frame (m)
  velocity_cam: np.ndarray  # [vX, vY, vZ] camera frame (m/s)
  position_ned: np.ndarray  # [N, E, D] NED frame (m)
  threat_level: str         # "safe" / "warning" / "critical"


class PositionSolver:
  """Compute target bearing, range, velocity and threat level.

  Transforms tracked target states into actionable output:
  azimuth, elevation, slant range, relative velocity, and
  time-to-collision (TTC).
  """

  def __init__(self, stereo_camera=None, config=None):
    """Initialize position solver.

    Args:
      stereo_camera: StereoCamera instance (for cam2body xform).
      config:        Configuration dict.
    """
    if config is None:
      config = load_config()
    self.config = config

    ps_cfg = config.get('position_solver', {})
    self.ttc_warn = ps_cfg.get('ttc_warning_threshold', 5.0)
    self.ttc_crit = ps_cfg.get('ttc_critical_threshold', 2.0)

    # Camera-to-body transformation
    if stereo_camera is not None:
      self.R_cam2body = stereo_camera.R_cam2body
      self.T_cam2body = stereo_camera.T_cam2body
    else:
      cam_cfg = config.get('camera', {}).get('cam_to_body', {})
      rot = cam_cfg.get('rotation', [0, 0, 0])
      self.R_cam2body = euler_to_rotation_matrix(
        rot[2], rot[1], rot[0]
      )
      self.T_cam2body = np.array(
        cam_cfg.get('translation', [0, 0, 0]),
        dtype=np.float64
      )

  def solve(self, track_states, imu_attitude=None):
    """Solve positions for all tracked targets.

    Args:
      track_states: List of TrackState objects from tracker.
      imu_attitude: [roll, pitch, yaw] in radians (optional).
                    If None, assumes level flight.

    Returns:
      List[TargetReport]: Resolved target reports.
    """
    if imu_attitude is None:
      imu_attitude = np.zeros(3)

    reports = []
    for ts in track_states:
      report = self._solve_single(ts, imu_attitude)
      if report is not None:
        reports.append(report)

    return reports

  def _solve_single(self, track_state, imu_attitude):
    """Solve position for a single track.

    Args:
      track_state: TrackState object.
      imu_attitude: [roll, pitch, yaw] radians.

    Returns:
      TargetReport or None if invalid.
    """
    pos_cam = track_state.position.copy()
    vel_cam = track_state.velocity.copy()

    # Slant range
    slant_range = np.linalg.norm(pos_cam)
    if slant_range < 0.1:
      return None

    # Azimuth: angle from forward (Z) axis in XZ plane
    # Positive = right, Negative = left
    azimuth_rad = np.arctan2(pos_cam[0], pos_cam[2])
    azimuth_deg = np.degrees(azimuth_rad)

    # Elevation: angle from forward (Z) axis in YZ plane
    # Positive = up (camera Y is down, so negate)
    elevation_rad = np.arctan2(-pos_cam[1], pos_cam[2])
    elevation_deg = np.degrees(elevation_rad)

    # Relative speed (magnitude)
    velocity_mps = np.linalg.norm(vel_cam)

    # Closing speed (radial component toward ego)
    # Positive = approaching, Negative = receding
    unit_range = pos_cam / slant_range
    closing_speed = -np.dot(vel_cam, unit_range)

    # Time-to-collision
    if closing_speed > 0.1:
      ttc = slant_range / closing_speed
    else:
      ttc = float('inf')  # Not approaching

    # Threat level
    threat = self._assess_threat(ttc)

    # Transform to NED frame
    pos_body = camera_to_body(
      pos_cam, self.R_cam2body, self.T_cam2body
    )
    roll, pitch, yaw = imu_attitude
    pos_ned = body_to_ned(pos_body, roll, pitch, yaw)

    return TargetReport(
      track_id=track_state.track_id,
      azimuth_deg=azimuth_deg,
      elevation_deg=elevation_deg,
      slant_range_m=slant_range,
      velocity_mps=velocity_mps,
      closing_speed_mps=closing_speed,
      ttc_sec=ttc,
      position_cam=pos_cam,
      velocity_cam=vel_cam,
      position_ned=pos_ned,
      threat_level=threat,
    )

  def _assess_threat(self, ttc):
    """Assess threat level based on TTC.

    Args:
      ttc: Time-to-collision in seconds.

    Returns:
      str: "safe", "warning", or "critical".
    """
    if ttc <= self.ttc_crit:
      return "critical"
    elif ttc <= self.ttc_warn:
      return "warning"
    else:
      return "safe"

  @staticmethod
  def compute_azimuth(pos_cam):
    """Compute azimuth angle from camera-frame position.

    Args:
      pos_cam: [X, Y, Z] in camera frame.

    Returns:
      float: Azimuth in degrees.
    """
    return np.degrees(np.arctan2(pos_cam[0], pos_cam[2]))

  @staticmethod
  def compute_elevation(pos_cam):
    """Compute elevation angle from camera-frame position.

    Args:
      pos_cam: [X, Y, Z] in camera frame.

    Returns:
      float: Elevation in degrees.
    """
    return np.degrees(np.arctan2(-pos_cam[1], pos_cam[2]))

  @staticmethod
  def compute_slant_range(pos_cam):
    """Compute slant range.

    Args:
      pos_cam: [X, Y, Z] in camera frame.

    Returns:
      float: Slant range in meters.
    """
    return float(np.linalg.norm(pos_cam))

  @staticmethod
  def compute_ttc(pos_cam, vel_cam):
    """Compute time-to-collision.

    Args:
      pos_cam: [X, Y, Z] position in camera frame.
      vel_cam: [vX, vY, vZ] velocity in camera frame.

    Returns:
      float: TTC in seconds. inf if not approaching.
    """
    r = np.linalg.norm(pos_cam)
    if r < 0.1:
      return 0.0
    unit_r = pos_cam / r
    closing = -np.dot(vel_cam, unit_r)
    if closing > 0.1:
      return r / closing
    return float('inf')
