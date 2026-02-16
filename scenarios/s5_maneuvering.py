# =============================================================================
# Scenario S5: Maneuvering Target
# Target performs aggressive evasive maneuvers.
# =============================================================================

import numpy as np
from scenarios.scenario_base import (
  ScenarioBase, TargetTrajectory,
  generate_linear_trajectory,
  generate_sinusoidal_maneuver,
)


class ManeuveringScenario(ScenarioBase):
  """S5: Maneuvering target scenario.

  Ego flies north at 200 km/h.
  Target approaches then performs sharp turns and altitude changes.
  Tests EKF prediction robustness under high maneuver.
  """

  def __init__(self, duration=10.0, frame_rate=120):
    super().__init__(duration, frame_rate)
    self.ego_speed = 200 / 3.6

  def get_name(self):
    return "S5_Maneuvering"

  def get_description(self):
    return ("Maneuvering target: sharp turns and altitude "
            "changes to test EKF robustness")

  def get_ego_trajectory(self):
    return generate_linear_trajectory(
      start_pos=[0, 0, -500],
      velocity=[self.ego_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=-1
    )

  def get_target_trajectories(self):
    # Target with sinusoidal lateral maneuver
    # Approaching from ahead with S-turns
    tgt = generate_sinusoidal_maneuver(
      start_pos=[1500, 0, -500],
      base_velocity=[-200 / 3.6, 0, 0],  # approaching
      amplitude=200.0,    # 200m lateral oscillation
      period=4.0,         # 4-second period
      maneuver_axis=1,    # East-West oscillation
      timestamps=self.timestamps,
      target_id=1
    )

    # Add vertical maneuver component
    omega_v = 2 * np.pi / 3.0  # 3-second vertical period
    tgt.positions[:, 2] += (
      100.0 * np.sin(omega_v * self.timestamps)
    )
    tgt.velocities[:, 2] += (
      100.0 * omega_v * np.cos(omega_v * self.timestamps)
    )

    return [tgt]
