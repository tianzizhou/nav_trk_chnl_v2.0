# 合成场景生成器 — 使用手册

## 1. 快速入门

```python
from src.camera_model import StereoCamera
from src.scene_generator import SceneGenerator
from scenarios.s1_head_on import HeadOnScenario

# Setup
camera = StereoCamera()
generator = SceneGenerator(camera)
scenario = HeadOnScenario(duration=5.0)

# Generate one frame
frame = generator.generate_frame(scenario, frame_idx=0)
print(f"Targets: {len(frame.ground_truth)}")
print(f"Image shape: {frame.left_image.shape}")
```

## 2. 可用场景列表

| 类名 | 场景 | 模块 |
|------|------|------|
| `HeadOnScenario` | S1 正面迎头 | `scenarios.s1_head_on` |
| `HighSpeedCrossScenario` | S2 高速交叉 | `scenarios.s2_high_speed_cross` |
| `TailChaseScenario` | S3 尾追 | `scenarios.s3_tail_chase` |
| `MultiTargetScenario` | S4 多目标 | `scenarios.s4_multi_target` |
| `ManeuveringScenario` | S5 机动规避 | `scenarios.s5_maneuvering` |
| `LongRangeScenario` | S6 远距离 | `scenarios.s6_long_range` |
| `ClutterScenario` | S7 杂波 | `scenarios.s7_clutter` |
| `OcclusionScenario` | S8 遮挡 | `scenarios.s8_occlusion` |

## 3. API 参考

### SceneGenerator

```python
gen = SceneGenerator(stereo_camera, config=None)

# Single frame
frame = gen.generate_frame(scenario, frame_idx)

# Sequence (generator)
for frame in gen.generate_sequence(scenario):
    process(frame)
```

### FrameData 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | float | 时间戳 (s) |
| `frame_idx` | int | 帧编号 |
| `left_image` | ndarray[H,W,3] | 左图 uint8 |
| `right_image` | ndarray[H,W,3] | 右图 uint8 |
| `ground_truth` | List[TargetGT] | GT列表 |
| `imu_attitude` | ndarray[3] | [roll,pitch,yaw] |

## 4. 创建自定义场景

```python
from scenarios.scenario_base import (
  ScenarioBase, generate_linear_trajectory
)

class MyScenario(ScenarioBase):
  def get_name(self):
    return "My_Custom"

  def get_description(self):
    return "Custom scenario"

  def get_ego_trajectory(self):
    return generate_linear_trajectory(
      [0, 0, -500], [50, 0, 0], self.timestamps, -1
    )

  def get_target_trajectories(self):
    return [generate_linear_trajectory(
      [1000, 0, -500], [-80, 0, 0], self.timestamps, 1
    )]
```

## 5. 常见问题

**Q: 如何禁用图像噪声？**
A: 在配置文件中设置 `simulation.image_synthesis.noise_std: 0`

**Q: 如何修改目标物理尺寸？**
A: 修改 `detection.target_size` 配置项。
