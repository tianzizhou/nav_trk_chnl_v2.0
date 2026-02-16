# 合成场景生成器 — 详细设计文档

## 1. 模块架构

```
scenarios/scenario_base.py       src/scene_generator.py
┌───────────────────────┐       ┌──────────────────────┐
│ ScenarioBase (ABC)     │       │ SceneGenerator        │
│ ├─ get_name()          │       │ ├─ generate_frame()   │
│ ├─ get_description()   │◄──────│ ├─ generate_sequence()│
│ ├─ get_ego_trajectory()│       │ ├─ _compute_bbox()    │
│ ├─ get_target_trajs()  │       │ └─ _render_image()    │
│ └─ get_ego_attitude()  │       └──────────────────────┘
├───────────────────────┤
│ TargetTrajectory       │       Data Classes:
│ ├─ target_id           │       ┌──────────────────────┐
│ ├─ positions  [N,3]    │       │ TargetGT              │
│ ├─ velocities [N,3]    │       │ ├─ target_id          │
│ └─ timestamps [N]      │       │ ├─ position_cam [3]   │
└───────────────────────┘       │ ├─ velocity_cam [3]   │
                                │ ├─ bbox_left [4]      │
scenarios/s1~s8_*.py             │ ├─ distance           │
┌───────────────────────┐       │ └─ visible            │
│ HeadOnScenario         │       ├──────────────────────┤
│ HighSpeedCrossScenario │       │ FrameData             │
│ TailChaseScenario      │       │ ├─ left_image  [H,W,3]│
│ MultiTargetScenario    │       │ ├─ right_image [H,W,3]│
│ ManeuveringScenario    │       │ ├─ ground_truth       │
│ LongRangeScenario      │       │ └─ imu_attitude [3]   │
│ ClutterScenario        │       └──────────────────────┘
│ OcclusionScenario      │
└───────────────────────┘
```

## 2. 核心算法

### 2.1 NED → Camera 坐标变换

```
R_ned2cam = R_cam2body^T * R_body2ned^T
T_ned2cam = R_cam2body^T * (-T_cam2body - R_ned2body^T * ego_pos)
p_cam = R_ned2cam * p_ned + T_ned2cam
```

### 2.2 检测框计算

```
w_px = wingspan * fx / distance
h_px = height * fy / distance
bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
```

### 2.3 图像合成流程

1. 绘制天空梯度背景
2. 对每个可见目标，绘制椭圆 + 翼展线
3. 叠加高斯噪声 (σ = noise_std)

## 3. 场景参数表

| 场景 | 己方速度 | 目标速度 | 初始距离 | 目标数 | 持续时间 |
|------|----------|----------|----------|--------|----------|
| S1 | 200km/h N | 300km/h S | 2000m | 1 | 10s |
| S2 | 300km/h N | 900km/h W | 1118m | 1 | 8s |
| S3 | 300km/h N | 200km/h N | 1500m | 1 | 15s |
| S4 | 200km/h N | 多方向 | 800~2000m | 5 | 10s |
| S5 | 200km/h N | 200km/h S+机动 | 1500m | 1 | 10s |
| S6 | 200km/h N | ~192km/h N | 1869m | 1 | 15s |
| S7 | 200km/h N | 150km/h SW | 1020m | 1 | 10s |
| S8 | 200km/h N | 200km/h SW×2 | 1044m | 2 | 10s |

## 4. Ground Truth 数据格式

每帧每目标的 `TargetGT` 包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| target_id | int | 目标唯一ID |
| position_cam | ndarray[3] | 相机坐标系位置 (m) |
| velocity_cam | ndarray[3] | 相机坐标系速度 (m/s) |
| position_ned | ndarray[3] | NED相对位置 (m) |
| velocity_ned | ndarray[3] | NED相对速度 (m/s) |
| bbox_left | ndarray[4] | 左图检测框 [x1,y1,x2,y2] |
| bbox_right | ndarray[4] | 右图检测框 |
| in_frame | bool | 是否在视场内 |
| distance | float | 斜距 (m) |
| visible | bool | 是否被遮挡 |
