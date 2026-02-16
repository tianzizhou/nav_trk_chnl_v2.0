# 位置解算模块 — 详细设计文档

## 1. PositionSolver 类

```python
class PositionSolver:
  def __init__(self, stereo_camera=None, config=None)
  def solve(self, track_states, imu_attitude=None)
      -> List[TargetReport]
```

## 2. TargetReport 数据结构

| 字段 | 类型 | 说明 |
|------|------|------|
| track_id | int | 跟踪ID |
| azimuth_deg | float | 方位角(度) |
| elevation_deg | float | 俯仰角(度) |
| slant_range_m | float | 斜距(米) |
| velocity_mps | float | 相对速度(m/s) |
| closing_speed_mps | float | 径向接近速度(m/s) |
| ttc_sec | float | TTC(秒), inf=远离 |
| position_cam | ndarray[3] | 相机坐标位置 |
| velocity_cam | ndarray[3] | 相机坐标速度 |
| position_ned | ndarray[3] | NED坐标位置 |
| threat_level | str | safe/warning/critical |

## 3. 公式

方位角: α = atan2(X_cam, Z_cam)
俯仰角: β = atan2(-Y_cam, Z_cam)
斜距: R = sqrt(X² + Y² + Z²)
径向接近速度: V_r = -V · (P/|P|)
TTC: T = R / V_r
