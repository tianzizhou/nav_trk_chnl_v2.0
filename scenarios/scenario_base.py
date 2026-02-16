# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved.
# Project Name : T2031
# Module Name  : scenario_base
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Features include:
# 1. Abstract base class for simulation scenarios
# 2. Common trajectory generation utilities
# =============================================================================

import numpy as np
from abc import ABC, abstractmethod


class TargetTrajectory:
  """Trajectory data for a single target drone.

  Attributes:
    target_id:    Unique identifier for the target.
    positions:    Nx3 array of 3D positions [x, y, z] in world frame.
    velocities:   Nx3 array of velocities [vx, vy, vz] in m/s.
    timestamps:   N-element array of timestamps in seconds.
    size:         Physical size dict {wingspan, height, length} in m.
  """

  def __init__(self, target_id, positions, velocities,
               timestamps, size=None):
    self.target_id = int(target_id)
    self.positions = np.asarray(positions, dtype=np.float64)
    self.velocities = np.asarray(velocities, dtype=np.float64)
    self.timestamps = np.asarray(timestamps, dtype=np.float64)
    self.size = size or {
      'wingspan': 1.2, 'height': 0.4, 'length': 1.0
    }


class ScenarioBase(ABC):
  """Abstract base class for simulation scenarios.

  All scenario implementations must provide ego and target
  trajectories expressed in the NED (North-East-Down) world frame.
  The pipeline will transform them into the camera frame.
  """

  def __init__(self, duration=10.0, frame_rate=120):
    """Initialize scenario.

    Args:
      duration:   Simulation duration in seconds.
      frame_rate: Frame rate in Hz.
    """
    self.duration = float(duration)
    self.frame_rate = int(frame_rate)
    self.num_frames = int(self.duration * self.frame_rate)
    self.dt = 1.0 / self.frame_rate
    self.timestamps = np.arange(self.num_frames) * self.dt

  @abstractmethod
  def get_name(self):
    """Return scenario name string."""
    pass

  @abstractmethod
  def get_description(self):
    """Return scenario description string."""
    pass

  @abstractmethod
  def get_ego_trajectory(self):
    """Return ego (own) drone trajectory.

    Returns:
      TargetTrajectory: Ego drone trajectory in NED frame.
        positions:  Nx3 [N, E, D] in meters
        velocities: Nx3 [vN, vE, vD] in m/s
    """
    pass

  @abstractmethod
  def get_target_trajectories(self):
    """Return list of target drone trajectories.

    Returns:
      list[TargetTrajectory]: Each in NED frame.
    """
    pass

  def get_ego_attitude(self):
    """Return ego drone attitude over time.

    Default: level flight heading north (yaw=0, pitch=0, roll=0).

    Returns:
      np.ndarray: Nx3 array of [roll, pitch, yaw] in radians.
    """
    return np.zeros((self.num_frames, 3), dtype=np.float64)


# ---- Trajectory generation utilities ----

def generate_linear_trajectory(start_pos, velocity, timestamps,
                               target_id=0, size=None):
  """Generate a constant-velocity linear trajectory.

  Args:
    start_pos:  [x, y, z] initial position (NED, meters).
    velocity:   [vx, vy, vz] constant velocity (m/s).
    timestamps: Array of timestamps (seconds).
    target_id:  Target identifier.
    size:       Physical size dict.

  Returns:
    TargetTrajectory: Linear trajectory.
  """
  start = np.asarray(start_pos, dtype=np.float64)
  vel = np.asarray(velocity, dtype=np.float64)
  ts = np.asarray(timestamps, dtype=np.float64)

  positions = start[None, :] + vel[None, :] * ts[:, None]
  velocities = np.tile(vel, (len(ts), 1))

  return TargetTrajectory(
    target_id, positions, velocities, ts, size
  )


def generate_accelerating_trajectory(start_pos, start_vel,
                                     acceleration, timestamps,
                                     target_id=0, size=None):
  """Generate a constant-acceleration trajectory.

  Args:
    start_pos:    [x, y, z] initial position (m).
    start_vel:    [vx, vy, vz] initial velocity (m/s).
    acceleration: [ax, ay, az] constant acceleration (m/s^2).
    timestamps:   Array of timestamps.
    target_id:    Target identifier.
    size:         Physical size dict.

  Returns:
    TargetTrajectory: Accelerating trajectory.
  """
  p0 = np.asarray(start_pos, dtype=np.float64)
  v0 = np.asarray(start_vel, dtype=np.float64)
  a = np.asarray(acceleration, dtype=np.float64)
  ts = np.asarray(timestamps, dtype=np.float64)

  positions = (p0[None, :]
               + v0[None, :] * ts[:, None]
               + 0.5 * a[None, :] * (ts[:, None] ** 2))
  velocities = v0[None, :] + a[None, :] * ts[:, None]

  return TargetTrajectory(
    target_id, positions, velocities, ts, size
  )


def generate_sinusoidal_maneuver(start_pos, base_velocity,
                                 amplitude, period,
                                 maneuver_axis, timestamps,
                                 target_id=0, size=None):
  """Generate a sinusoidal maneuvering trajectory.

  The target flies with a base velocity plus sinusoidal
  oscillation on a specified axis.

  Args:
    start_pos:     [x, y, z] initial position (m).
    base_velocity: [vx, vy, vz] mean velocity (m/s).
    amplitude:     Oscillation amplitude (meters).
    period:        Oscillation period (seconds).
    maneuver_axis: 0=N, 1=E, 2=D axis for oscillation.
    timestamps:    Array of timestamps.
    target_id:     Target identifier.
    size:          Physical size dict.

  Returns:
    TargetTrajectory: Maneuvering trajectory.
  """
  p0 = np.asarray(start_pos, dtype=np.float64)
  v0 = np.asarray(base_velocity, dtype=np.float64)
  ts = np.asarray(timestamps, dtype=np.float64)
  omega = 2 * np.pi / period

  # Base linear motion + sinusoidal offset
  positions = p0[None, :] + v0[None, :] * ts[:, None]
  positions[:, maneuver_axis] += (
    amplitude * np.sin(omega * ts)
  )

  # Velocity: base + sinusoidal derivative
  velocities = np.tile(v0, (len(ts), 1))
  velocities[:, maneuver_axis] += (
    amplitude * omega * np.cos(omega * ts)
  )

  return TargetTrajectory(
    target_id, positions, velocities, ts, size
  )
