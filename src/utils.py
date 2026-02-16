# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved. No part of this code may be reproduced, modified, or distributed
# without the written permission of Shanghai Wuben Technology Co., Ltd.
# Project Name : T2031
# Module Name  : utils
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Version History:
# 1.0.0 - 2026-02-16: Initial version.
#
# Features include:
# 1. Coordinate frame transformations (pixel/camera/body/NED)
# 2. Rotation representation conversions (Euler/matrix/quaternion)
# 3. Mathematical utility functions
#
# Notes:
# 1. This code is for internal use only.
# =============================================================================

import numpy as np
import yaml


def load_config(config_path="config/default_config.yaml"):
  """Load configuration from YAML file.

  Args:
    config_path: Path to the YAML configuration file.

  Returns:
    dict: Configuration dictionary.
  """
  with open(config_path, 'r') as f:
    return yaml.safe_load(f)


# ---- Rotation representation conversions ----

def euler_to_rotation_matrix(roll, pitch, yaw):
  """Convert Euler angles (ZYX convention) to rotation matrix.

  Rotation order: first yaw (Z), then pitch (Y), then roll (X).

  Args:
    roll:  Roll angle in radians (rotation about X-axis).
    pitch: Pitch angle in radians (rotation about Y-axis).
    yaw:   Yaw angle in radians (rotation about Z-axis).

  Returns:
    np.ndarray: 3x3 rotation matrix.
  """
  cr = np.cos(roll)
  sr = np.sin(roll)
  cp = np.cos(pitch)
  sp = np.sin(pitch)
  cy = np.cos(yaw)
  sy = np.sin(yaw)

  # R = Rz(yaw) * Ry(pitch) * Rx(roll)
  r00 = cy * cp
  r01 = cy * sp * sr - sy * cr
  r02 = cy * sp * cr + sy * sr
  r10 = sy * cp
  r11 = sy * sp * sr + cy * cr
  r12 = sy * sp * cr - cy * sr
  r20 = -sp
  r21 = cp * sr
  r22 = cp * cr

  return np.array([
    [r00, r01, r02],
    [r10, r11, r12],
    [r20, r21, r22]
  ])


def rotation_matrix_to_euler(R):
  """Convert rotation matrix to Euler angles (ZYX convention).

  Args:
    R: 3x3 rotation matrix.

  Returns:
    tuple: (roll, pitch, yaw) in radians.
  """
  # Handle gimbal lock
  sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

  singular = sy < 1e-6

  if not singular:
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], sy)
    yaw = np.arctan2(R[1, 0], R[0, 0])
  else:
    roll = np.arctan2(-R[1, 2], R[1, 1])
    pitch = np.arctan2(-R[2, 0], sy)
    yaw = 0.0

  return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
  """Convert Euler angles (ZYX) to quaternion [w, x, y, z].

  Args:
    roll:  Roll angle in radians.
    pitch: Pitch angle in radians.
    yaw:   Yaw angle in radians.

  Returns:
    np.ndarray: Quaternion [w, x, y, z].
  """
  cr = np.cos(roll / 2)
  sr = np.sin(roll / 2)
  cp = np.cos(pitch / 2)
  sp = np.sin(pitch / 2)
  cy = np.cos(yaw / 2)
  sy = np.sin(yaw / 2)

  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy

  return np.array([w, x, y, z])


def quaternion_to_euler(q):
  """Convert quaternion [w, x, y, z] to Euler angles (ZYX).

  Args:
    q: Quaternion array [w, x, y, z].

  Returns:
    tuple: (roll, pitch, yaw) in radians.
  """
  w, x, y, z = q[0], q[1], q[2], q[3]

  # Roll (X-axis rotation)
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = np.arctan2(sinr_cosp, cosr_cosp)

  # Pitch (Y-axis rotation)
  sinp = 2.0 * (w * y - z * x)
  sinp = np.clip(sinp, -1.0, 1.0)
  pitch = np.arcsin(sinp)

  # Yaw (Z-axis rotation)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = np.arctan2(siny_cosp, cosy_cosp)

  return roll, pitch, yaw


def quaternion_to_rotation_matrix(q):
  """Convert quaternion [w, x, y, z] to rotation matrix.

  Args:
    q: Quaternion array [w, x, y, z].

  Returns:
    np.ndarray: 3x3 rotation matrix.
  """
  w, x, y, z = q[0], q[1], q[2], q[3]
  n = w * w + x * x + y * y + z * z
  s = 0.0 if n < 1e-12 else 2.0 / n

  wx = s * w * x
  wy = s * w * y
  wz = s * w * z
  xx = s * x * x
  xy = s * x * y
  xz = s * x * z
  yy = s * y * y
  yz = s * y * z
  zz = s * z * z

  return np.array([
    [1 - yy - zz, xy - wz, xz + wy],
    [xy + wz, 1 - xx - zz, yz - wx],
    [xz - wy, yz + wx, 1 - xx - yy]
  ])


def rotation_matrix_to_quaternion(R):
  """Convert rotation matrix to quaternion [w, x, y, z].

  Uses Shepperd's method for numerical stability.

  Args:
    R: 3x3 rotation matrix.

  Returns:
    np.ndarray: Quaternion [w, x, y, z].
  """
  trace = R[0, 0] + R[1, 1] + R[2, 2]

  if trace > 0:
    s = 0.5 / np.sqrt(trace + 1.0)
    w = 0.25 / s
    x = (R[2, 1] - R[1, 2]) * s
    y = (R[0, 2] - R[2, 0]) * s
    z = (R[1, 0] - R[0, 1]) * s
  elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
    w = (R[2, 1] - R[1, 2]) / s
    x = 0.25 * s
    y = (R[0, 1] + R[1, 0]) / s
    z = (R[0, 2] + R[2, 0]) / s
  elif R[1, 1] > R[2, 2]:
    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
    w = (R[0, 2] - R[2, 0]) / s
    x = (R[0, 1] + R[1, 0]) / s
    y = 0.25 * s
    z = (R[1, 2] + R[2, 1]) / s
  else:
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    w = (R[1, 0] - R[0, 1]) / s
    x = (R[0, 2] + R[2, 0]) / s
    y = (R[1, 2] + R[2, 1]) / s
    z = 0.25 * s

  return np.array([w, x, y, z])


def rodrigues_to_rotation_matrix(rvec):
  """Convert Rodrigues rotation vector to rotation matrix.

  Args:
    rvec: Rotation vector (3,) — direction = axis, norm = angle.

  Returns:
    np.ndarray: 3x3 rotation matrix.
  """
  rvec = np.asarray(rvec, dtype=np.float64).flatten()
  theta = np.linalg.norm(rvec)

  if theta < 1e-12:
    return np.eye(3)

  k = rvec / theta
  K = np.array([
    [0, -k[2], k[1]],
    [k[2], 0, -k[0]],
    [-k[1], k[0], 0]
  ])

  # Rodrigues' formula: R = I + sin(t)*K + (1-cos(t))*K^2
  R = (np.eye(3)
       + np.sin(theta) * K
       + (1 - np.cos(theta)) * (K @ K))
  return R


# ---- Coordinate frame transformations ----

def camera_to_body(p_cam, R_cam2body, T_cam2body):
  """Transform point from camera frame to body frame.

  Args:
    p_cam:      3D point in camera frame [X, Y, Z].
    R_cam2body: 3x3 rotation matrix (camera -> body).
    T_cam2body: 3D translation vector (camera -> body).

  Returns:
    np.ndarray: 3D point in body frame.
  """
  p_cam = np.asarray(p_cam, dtype=np.float64).flatten()
  T_cam2body = np.asarray(T_cam2body, dtype=np.float64).flatten()
  return R_cam2body @ p_cam + T_cam2body


def body_to_camera(p_body, R_cam2body, T_cam2body):
  """Transform point from body frame to camera frame.

  Args:
    p_body:     3D point in body frame.
    R_cam2body: 3x3 rotation matrix (camera -> body).
    T_cam2body: 3D translation vector (camera -> body).

  Returns:
    np.ndarray: 3D point in camera frame.
  """
  p_body = np.asarray(p_body, dtype=np.float64).flatten()
  T_cam2body = np.asarray(T_cam2body, dtype=np.float64).flatten()
  return R_cam2body.T @ (p_body - T_cam2body)


def body_to_ned(p_body, roll, pitch, yaw):
  """Transform point from body frame to NED frame.

  Args:
    p_body: 3D point in body frame.
    roll:   Roll angle (radians).
    pitch:  Pitch angle (radians).
    yaw:    Yaw angle (radians).

  Returns:
    np.ndarray: 3D point in NED frame.
  """
  R = euler_to_rotation_matrix(roll, pitch, yaw)
  p_body = np.asarray(p_body, dtype=np.float64).flatten()
  return R @ p_body


def ned_to_body(p_ned, roll, pitch, yaw):
  """Transform point from NED frame to body frame.

  Args:
    p_ned:  3D point in NED frame.
    roll:   Roll angle (radians).
    pitch:  Pitch angle (radians).
    yaw:    Yaw angle (radians).

  Returns:
    np.ndarray: 3D point in body frame.
  """
  R = euler_to_rotation_matrix(roll, pitch, yaw)
  p_ned = np.asarray(p_ned, dtype=np.float64).flatten()
  return R.T @ p_ned


# ---- Mathematical utilities ----

def normalize_angle(angle):
  """Normalize angle to [-pi, pi].

  Args:
    angle: Angle in radians.

  Returns:
    float: Normalized angle in [-pi, pi].
  """
  return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_angle_deg(angle):
  """Normalize angle to [-180, 180] degrees.

  Args:
    angle: Angle in degrees.

  Returns:
    float: Normalized angle in [-180, 180].
  """
  return (angle + 180.0) % 360.0 - 180.0


def unit_vector(v):
  """Compute unit vector.

  Args:
    v: Input vector.

  Returns:
    np.ndarray: Unit vector, or zero vector if input is zero.
  """
  v = np.asarray(v, dtype=np.float64)
  n = np.linalg.norm(v)
  if n < 1e-12:
    return np.zeros_like(v)
  return v / n


def skew_symmetric(v):
  """Compute skew-symmetric matrix from 3D vector.

  Args:
    v: 3D vector [x, y, z].

  Returns:
    np.ndarray: 3x3 skew-symmetric matrix.
  """
  v = np.asarray(v, dtype=np.float64).flatten()
  return np.array([
    [0, -v[2], v[1]],
    [v[2], 0, -v[0]],
    [-v[1], v[0], 0]
  ])
