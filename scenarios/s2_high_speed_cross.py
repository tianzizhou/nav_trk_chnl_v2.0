# =============================================================================
# Scenario S2: High-Speed Crossing
# Target crosses from the side at very high relative speed.
# =============================================================================

from scenarios.scenario_base import (
  ScenarioBase, generate_linear_trajectory,
)


class HighSpeedCrossScenario(ScenarioBase):
  """S2: High-speed crossing scenario.

  Ego flies north at 300 km/h.
  Target flies west at 900 km/h, crossing from east to west.
  Relative speed: ~950 km/h (vector sum).
  """

  def __init__(self, duration=8.0, frame_rate=120):
    super().__init__(duration, frame_rate)
    self.ego_speed = 300 / 3.6     # ~83.3 m/s
    self.target_speed = 900 / 3.6  # ~250 m/s

  def get_name(self):
    return "S2_High_Speed_Cross"

  def get_description(self):
    return ("High-speed crossing: target crosses from east "
            "at ~1200 km/h relative speed")

  def get_ego_trajectory(self):
    return generate_linear_trajectory(
      start_pos=[0, 0, -500],
      velocity=[self.ego_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=-1
    )

  def get_target_trajectories(self):
    # Target starts 1km east, 500m ahead, same altitude
    # Flies west (negative E direction)
    tgt = generate_linear_trajectory(
      start_pos=[500, 1000, -500],
      velocity=[0, -self.target_speed, 0],
      timestamps=self.timestamps,
      target_id=1
    )
    return [tgt]
