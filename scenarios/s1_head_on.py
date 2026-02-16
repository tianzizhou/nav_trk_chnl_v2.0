# =============================================================================
# Scenario S1: Head-On Approach
# Target approaches from directly ahead at high closing speed.
# =============================================================================

from scenarios.scenario_base import (
  ScenarioBase, TargetTrajectory,
  generate_linear_trajectory,
)


class HeadOnScenario(ScenarioBase):
  """S1: Head-on approach scenario.

  Ego flies north at 200 km/h.
  Target flies south (toward ego) at 300 km/h.
  Closing speed: 500 km/h (139 m/s).
  Initial separation: 2000m.
  """

  def __init__(self, duration=10.0, frame_rate=120):
    super().__init__(duration, frame_rate)
    self.ego_speed = 200 / 3.6     # m/s (~55.6)
    self.target_speed = 300 / 3.6  # m/s (~83.3)
    self.initial_range = 2000.0    # meters

  def get_name(self):
    return "S1_Head_On"

  def get_description(self):
    return ("Head-on approach: target from 2km ahead, "
            "closing speed ~500 km/h")

  def get_ego_trajectory(self):
    # Ego flies north (positive N direction)
    return generate_linear_trajectory(
      start_pos=[0, 0, -500],            # 500m altitude
      velocity=[self.ego_speed, 0, 0],   # heading north
      timestamps=self.timestamps,
      target_id=-1
    )

  def get_target_trajectories(self):
    # Target flies south (negative N) starting 2km north
    tgt = generate_linear_trajectory(
      start_pos=[self.initial_range, 0, -500],
      velocity=[-self.target_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=1
    )
    return [tgt]
