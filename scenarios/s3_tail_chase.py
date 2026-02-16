# =============================================================================
# Scenario S3: Tail Chase
# Ego chases a slower target from behind.
# =============================================================================

from scenarios.scenario_base import (
  ScenarioBase, generate_linear_trajectory,
)


class TailChaseScenario(ScenarioBase):
  """S3: Tail chase scenario.

  Ego flies north at 300 km/h.
  Target flies north at 200 km/h, starting 1500m ahead.
  Closing speed: 100 km/h (27.8 m/s).
  This tests long-range small target detection.
  """

  def __init__(self, duration=15.0, frame_rate=120):
    super().__init__(duration, frame_rate)
    self.ego_speed = 300 / 3.6
    self.target_speed = 200 / 3.6
    self.initial_range = 1500.0

  def get_name(self):
    return "S3_Tail_Chase"

  def get_description(self):
    return ("Tail chase: ego pursues target 1500m ahead, "
            "closing speed ~100 km/h")

  def get_ego_trajectory(self):
    return generate_linear_trajectory(
      start_pos=[0, 0, -500],
      velocity=[self.ego_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=-1
    )

  def get_target_trajectories(self):
    tgt = generate_linear_trajectory(
      start_pos=[self.initial_range, 0, -500],
      velocity=[self.target_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=1
    )
    return [tgt]
