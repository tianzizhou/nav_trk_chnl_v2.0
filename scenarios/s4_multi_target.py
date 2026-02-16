# =============================================================================
# Scenario S4: Multi-Target Swarm
# Multiple targets approach from different directions.
# =============================================================================

from scenarios.scenario_base import (
  ScenarioBase, generate_linear_trajectory,
)


class MultiTargetScenario(ScenarioBase):
  """S4: Multi-target swarm scenario.

  Ego flies north at 200 km/h.
  5 targets approach from different directions at various speeds.
  """

  def __init__(self, duration=10.0, frame_rate=120):
    super().__init__(duration, frame_rate)
    self.ego_speed = 200 / 3.6

  def get_name(self):
    return "S4_Multi_Target"

  def get_description(self):
    return ("Multi-target: 5 targets from different "
            "directions and distances")

  def get_ego_trajectory(self):
    return generate_linear_trajectory(
      start_pos=[0, 0, -500],
      velocity=[self.ego_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=-1
    )

  def get_target_trajectories(self):
    targets = []

    # Target 1: Head-on from north
    targets.append(generate_linear_trajectory(
      start_pos=[1500, 0, -500],
      velocity=[-250 / 3.6, 0, 0],
      timestamps=self.timestamps, target_id=1
    ))

    # Target 2: From north-east
    targets.append(generate_linear_trajectory(
      start_pos=[1000, 800, -480],
      velocity=[-200 / 3.6, -150 / 3.6, 0],
      timestamps=self.timestamps, target_id=2
    ))

    # Target 3: From north-west
    targets.append(generate_linear_trajectory(
      start_pos=[1200, -600, -520],
      velocity=[-180 / 3.6, 100 / 3.6, 0],
      timestamps=self.timestamps, target_id=3
    ))

    # Target 4: From above-ahead
    targets.append(generate_linear_trajectory(
      start_pos=[800, 200, -800],
      velocity=[-150 / 3.6, -50 / 3.6, 100 / 3.6],
      timestamps=self.timestamps, target_id=4
    ))

    # Target 5: Low and fast from ahead
    targets.append(generate_linear_trajectory(
      start_pos=[2000, -100, -400],
      velocity=[-300 / 3.6, 0, 0],
      timestamps=self.timestamps, target_id=5
    ))

    return targets
