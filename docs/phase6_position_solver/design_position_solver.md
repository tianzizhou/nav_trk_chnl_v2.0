# 位置解算模块 — 详细设计文档

## 1. 模块架构

```
PositionSolver
├─ solve(track_states, imu_attitude)
│   -> List[TargetReport]
├─ _solve_single(track_state, imu)
├─ _assess_threat(ttc)
├─ compute_azimuth(pos_cam)          [static]
├─ compute_elevation(pos_cam)        [static]
├─ compute_slant_range(pos_cam)      [static]
└─ compute_ttc(pos_cam, vel_cam)     [static]

TargetReport (dataclass)
├─ track_id: int
├─ azimuth_deg: float
├─ elevation_deg: float
├─ slant_range_m: float
├─ velocity_mps: float
├─ closing_speed_mps: float
├─ ttc_sec: float
├─ position_cam: ndarray[3]
├─ velocity_cam: ndarray[3]
├─ position_ned: ndarray[3]
└─ threat_level: str
```

---

## 2. 接口定义

### 2.1 PositionSolver 类

```python
class PositionSolver:
    def __init__(self, stereo_camera=None, config=None)
    def solve(self, track_states: List[TrackState],
              imu_attitude: Optional[np.ndarray] = None
              ) -> List[TargetReport]
```

### 2.2 TargetReport 数据结构

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `track_id` | int | — | 跟踪 ID |
| `azimuth_deg` | float | 度 | 方位角，+右/-左 |
| `elevation_deg` | float | 度 | 俯仰角，+上/-下 |
| `slant_range_m` | float | 米 | 斜距 |
| `velocity_mps` | float | m/s | 相对速度大小 |
| `closing_speed_mps` | float | m/s | 径向接近速度 |
| `ttc_sec` | float | 秒 | 碰撞预警时间 |
| `position_cam` | ndarray[3] | 米 | 相机坐标系位置 |
| `velocity_cam` | ndarray[3] | m/s | 相机坐标系速度 |
| `position_ned` | ndarray[3] | 米 | NED坐标系位置 |
| `threat_level` | str | — | safe/warning/critical |

---

## 3. 核心算法详细推导

### 3.1 方位角计算

相机坐标系定义：X→右，Y→下，Z→前。

方位角 α 定义为目标在 XZ 平面上相对 Z 轴的偏角：

```
α = atan2(X_cam, Z_cam)
```

- α = 0°：目标在正前方
- α = +45°：目标在右前方 45°
- α = -90°：目标在正左方

### 3.2 俯仰角计算

俯仰角 β 定义为目标在 YZ 平面上相对 Z 轴的偏角。
由于相机 Y 轴向下，取负号使向上为正：

```
β = atan2(-Y_cam, Z_cam)
```

- β = 0°：目标在水平方向
- β = +30°：目标在上方 30°
- β = -15°：目标在下方 15°

### 3.3 径向接近速度分解

相对位置向量 P，相对速度向量 V：

```
单位距离向量: u_r = P / |P|
径向速度分量: V_radial = V · u_r
接近速度: V_closing = -V_radial
```

V_closing > 0 表示目标正在接近。

### 3.4 坐标变换链推导

**Step 1: 相机 → 机体**

```
P_body = R_cam2body × P_cam + T_cam2body
```

R_cam2body 为标准置换矩阵（相机XYZ → 机体XYZ）：
```
Camera: X-right, Y-down, Z-forward
Body:   X-forward, Y-right, Z-down

R_cam2body = | 0  0  1 |   (cam_Z → body_X)
             | 1  0  0 |   (cam_X → body_Y)
             | 0  1  0 |   (cam_Y → body_Z)
```

**Step 2: 机体 → NED**

```
P_ned = R_body2ned × P_body
R_body2ned = R_z(ψ) × R_y(θ) × R_x(φ)
```

其中 φ=roll, θ=pitch, ψ=yaw 由 IMU 测量。

水平飞行（φ=θ=ψ=0）时 R_body2ned = I，机体坐标 = NED 坐标。

---

## 4. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `position_solver.ttc_warning_threshold` | 5.0 | 预警阈值 (s) |
| `position_solver.ttc_critical_threshold` | 2.0 | 危急阈值 (s) |
| `camera.cam_to_body.use_standard_transform` | true | 标准 cam→body |
| `camera.cam_to_body.translation` | [0.1, 0, -0.05] | 安装偏移 (m) |

---

## 5. 错误处理

| 条件 | 处理 |
|------|------|
| 斜距 < 0.1m | 返回 None（无效轨迹） |
| V_closing ≤ 0 | TTC = ∞（目标远离） |
| IMU 姿态为 None | 假设水平飞行 [0,0,0] |
