# =============================================================================
# Scenario S8: Occlusion / Track Recovery
# Targets temporarily disappear (occluded), testing track recovery.
# =============================================================================

import numpy as np
from scenarios.scenario_base import (
  ScenarioBase, TargetTrajectory,
  generate_linear_trajectory,
)


class OcclusionScenario(ScenarioBase):
  """S8: Occlusion and track recovery scenario.

  Ego flies north at 200 km/h.
  Two targets approach; they cross paths causing mutual
  occlusion (modeled as detection gaps). Tests tracker's
  ability to maintain and recover track IDs.
  """

  def __init__(self, duration=10.0, frame_rate=120):
    super().__init__(duration, frame_rate)
    self.ego_speed = 200 / 3.6

  def get_name(self):
    return "S8_Occlusion"

  def get_description(self):
    return ("Occlusion: two targets cross paths, causing "
            "temporary detection loss and track recovery test")

  def get_ego_trajectory(self):
    return generate_linear_trajectory(
      start_pos=[0, 0, -500],
      velocity=[self.ego_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=-1
    )

  def get_target_trajectories(self):
    targets = []

    # Target 1: approaching from north-east
    targets.append(generate_linear_trajectory(
      start_pos=[1000, 300, -500],
      velocity=[-200 / 3.6, -60 / 3.6, 0],
      timestamps=self.timestamps, target_id=1
    ))

    # Target 2: approaching from north-west
    # Paths will cross at roughly t=5s
    targets.append(generate_linear_trajectory(
      start_pos=[1000, -300, -500],
      velocity=[-200 / 3.6, 60 / 3.6, 0],
      timestamps=self.timestamps, target_id=2
    ))

    return targets

  def get_occlusion_intervals(self):
    """Return time intervals when targets are occluded.

    Returns:
      dict: {target_id: list of (start_time, end_time) tuples}
    """
    # Targets cross at ~t=5s, causing brief occlusion
    return {
      1: [(4.5, 5.5)],  # Target 1 lost for 1 second
      2: [(4.5, 5.5)],  # Target 2 lost for 1 second
    }
