# =============================================================================
# Scenario S7: Clutter / False Alarm
# Scenario with potential false alarm sources (modeled as extra targets
# that the detector might confuse with real drones).
# =============================================================================

from scenarios.scenario_base import (
  ScenarioBase, generate_linear_trajectory,
)


class ClutterScenario(ScenarioBase):
  """S7: Clutter and false alarm scenario.

  Ego flies north at 200 km/h.
  One real target plus simulated clutter sources.
  The detector's false alarm rate is tested here.
  Clutter is handled in the detector configuration, not as
  physical trajectories; only the real target is modeled.
  """

  def __init__(self, duration=10.0, frame_rate=120):
    super().__init__(duration, frame_rate)
    self.ego_speed = 200 / 3.6

  def get_name(self):
    return "S7_Clutter"

  def get_description(self):
    return ("Clutter: one real target with elevated "
            "false alarm rate to test FAR control")

  def get_ego_trajectory(self):
    return generate_linear_trajectory(
      start_pos=[0, 0, -500],
      velocity=[self.ego_speed, 0, 0],
      timestamps=self.timestamps,
      target_id=-1
    )

  def get_target_trajectories(self):
    # Single real target approaching
    tgt = generate_linear_trajectory(
      start_pos=[1000, 200, -500],
      velocity=[-150 / 3.6, -30 / 3.6, 0],
      timestamps=self.timestamps,
      target_id=1
    )
    return [tgt]

  def get_detector_overrides(self):
    """Return detector parameter overrides for this scenario.

    Increases false alarm rate to stress-test FAR filtering.
    """
    return {
      'false_alarm_rate': 0.05  # 5% FAR (10x normal)
    }
