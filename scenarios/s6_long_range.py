# =============================================================================
# Scenario S6: Long-Range Detection
# Target at extreme distance (1500-2000m), testing small target detection.
# =============================================================================

from scenarios.scenario_base import (
  ScenarioBase, generate_linear_trajectory,
)


class LongRangeScenario(ScenarioBase):
  """S6: Long-range detection scenario.

  Ego flies north at 200 km/h.
  Target flies parallel at similar speed, 1500-2000m away.
  Tests detection sensitivity for very small targets.
  """

  def __init__(self, duration=15.0, frame_rate=120):
    super().__init__(duration, frame_rate)
    self.ego_speed = 200 / 3.6

  def get_name(self):
    return "S6_Long_Range"

  def get_description(self):
    return ("Long-range detection: target at 1500-2000m "
            "distance, very small in image")

  def get_ego_trajectory(self):
    return generate_linear_trajectory(
      start_pos=[0, 0, -500],
      velocity=[self.ego_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=-1
    )

  def get_target_trajectories(self):
    # Target flies roughly parallel, slowly approaching
    # Start at 2000m range, closing at ~30 km/h
    tgt = generate_linear_trajectory(
      start_pos=[1800, 500, -520],
      velocity=[self.ego_speed - 30 / 3.6, -5 / 3.6, 0],
      timestamps=self.timestamps,
      target_id=1
    )
    return [tgt]
