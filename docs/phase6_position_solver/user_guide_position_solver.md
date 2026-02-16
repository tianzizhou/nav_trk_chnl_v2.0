# 位置解算模块 — 使用手册

## 1. 快速入门

```python
from src.position_solver import PositionSolver
from src.tracker import TrackState, TrackStatus
import numpy as np

solver = PositionSolver()

# Create a track state (normally from MultiTargetTracker)
track = TrackState(
    track_id=1,
    status=TrackStatus.CONFIRMED,
    position=np.array([100, -50, 500]),   # camera frame
    velocity=np.array([-10, 5, -100]),     # approaching
    acceleration=np.zeros(3),
    covariance=np.eye(9),
)

# Solve with level flight attitude
reports = solver.solve(
    [track], imu_attitude=np.array([0, 0, 0])
)

for r in reports:
    print(f"Target {r.track_id}:")
    print(f"  Azimuth:  {r.azimuth_deg:+.1f}°")
    print(f"  Elevation: {r.elevation_deg:+.1f}°")
    print(f"  Range:    {r.slant_range_m:.0f} m")
    print(f"  Speed:    {r.velocity_mps:.0f} m/s")
    print(f"  Closing:  {r.closing_speed_mps:.1f} m/s")
    print(f"  TTC:      {r.ttc_sec:.1f} s")
    print(f"  Threat:   {r.threat_level}")
    print(f"  NED:      {r.position_ned}")
```

---

## 2. 坐标系约定

```
相机坐标系 (Camera Frame)        机体坐标系 (Body Frame)
    Y (下)                           Z (下)
    |                                |
    |                                |
    +------ X (右)                   +------ Y (右)
   /                                /
  Z (前)                           X (前)

NED 坐标系 (水平飞行时 = 机体坐标系)
    D (下)
    |
    |
    +------ E (东)
   /
  N (北)
```

### 方位角/俯仰角定义

| 角度 | 参考 | 正方向 | 范围 |
|------|------|--------|------|
| 方位角 α | 相机 Z 轴（正前方） | 向右为正 | [-180°, +180°] |
| 俯仰角 β | 相机 Z 轴（水平） | 向上为正 | [-90°, +90°] |

---

## 3. API 参考

### 3.1 PositionSolver

```python
solver = PositionSolver(stereo_camera=None, config=None)
```

#### solve(track_states, imu_attitude)

| 参数 | 类型 | 说明 |
|------|------|------|
| `track_states` | List[TrackState] | 跟踪器输出的轨迹列表 |
| `imu_attitude` | ndarray[3] | [roll, pitch, yaw] rad, 可选 |
| **返回** | List[TargetReport] | 目标报告列表 |

### 3.2 静态工具方法

```python
az = PositionSolver.compute_azimuth(pos_cam)      # -> degrees
el = PositionSolver.compute_elevation(pos_cam)     # -> degrees
rng = PositionSolver.compute_slant_range(pos_cam)  # -> meters
ttc = PositionSolver.compute_ttc(pos_cam, vel_cam) # -> seconds
```

---

## 4. 输出示例（JSON 格式）

```json
{
  "track_id": 1,
  "azimuth_deg": 11.3,
  "elevation_deg": -5.7,
  "slant_range_m": 512.4,
  "velocity_mps": 101.2,
  "closing_speed_mps": 98.5,
  "ttc_sec": 5.2,
  "threat_level": "safe",
  "position_ned": [500.1, 100.2, 50.3]
}
```

---

## 5. 威胁等级说明

| 等级 | TTC 范围 | 显示颜色 | 含义 |
|------|---------|---------|------|
| `safe` | > 5s 或 ∞ | 🟢 绿色 | 无即时威胁 |
| `warning` | 2s ~ 5s | 🟡 黄色 | 准备规避 |
| `critical` | < 2s | 🔴 红色 | 立即规避 |

阈值可在配置文件中调整：
```yaml
position_solver:
  ttc_warning_threshold: 5.0
  ttc_critical_threshold: 2.0
```

---

## 6. 常见问题

**Q: TTC 显示为 inf 是什么意思？**

A: 目标正在远离（径向接近速度 ≤ 0），不存在碰撞风险。

**Q: 方位角 NED 如何换算？**

A: 方位角是相对相机前方的，如需地理方位角，
需加上己方航向角（yaw）：geo_az = azimuth + yaw。

**Q: 为什么 NED 坐标是相对位置？**

A: 系统不包含 GPS 绝对位置，NED 坐标是目标相对于己方的相对位置。
