# 双目摄像头无人机检测仿真系统 — 系统架构设计文档

## 1. 模块架构总览

### 1.1 系统模块划分

本系统由以下 8 个核心模块组成：

```
┌──────────────────────────────────────────────────────────────┐
│                     仿真控制层 (Simulation)                    │
│  run_simulation / batch_evaluation / report_generator         │
├──────────────────────────────────────────────────────────────┤
│                     流水线整合层 (Pipeline)                    │
│  DetectionPipeline: 单帧处理 / 序列处理 / 状态管理             │
├─────────────┬────────────┬────────────┬──────────────────────┤
│  检测模块    │ 立体视觉    │ 跟踪模块    │ 位置解算模块          │
│  Detector   │ Stereo     │ Tracker    │ PositionSolver       │
├─────────────┴────────────┴────────────┴──────────────────────┤
│                     场景生成层 (Scene Generator)               │
│  SceneGenerator + ScenarioBase + S1~S8 场景实现               │
├──────────────────────────────────────────────────────────────┤
│                     基础设施层 (Infrastructure)                │
│  StereoCamera (相机模型) + Utils (坐标变换/数学工具)           │
│  Config (YAML配置加载)                                        │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 模块职责

| 模块 | 源文件 | 职责 |
|------|--------|------|
| **StereoCamera** | `camera_model.py` | 双目相机内外参建模、投影/反投影、畸变处理 |
| **Utils** | `utils.py` | 坐标系变换、旋转表示转换、数学工具 |
| **SceneGenerator** | `scene_generator.py` | 生成合成场景（轨迹+图像+GT） |
| **Scenarios** | `scenarios/s1~s8` | 8 类飞行场景的参数化定义 |
| **StereoProcessor** | `stereo_processor.py` | 立体校正、SGBM视差、深度估计 |
| **Detector** | `detector.py` | 目标检测（模拟检测器+接口） |
| **Tracker** | `tracker.py` | EKF 跟踪 + 多目标管理 |
| **PositionSolver** | `position_solver.py` | 3D 位置/方位/速度/TTC 计算 |
| **Pipeline** | `pipeline.py` | 完整处理流水线整合 |
| **Simulation** | `sim/*.py` | 仿真运行/评估/报告 |

---

## 2. 数据流设计

### 2.1 帧级数据流

```
SceneGenerator.generate_frame(t)
  │
  ├─→ FrameData:
  │     left_image:  np.ndarray [H, W, 3] uint8
  │     right_image: np.ndarray [H, W, 3] uint8
  │     ground_truth: List[TargetGT]
  │       ├─ position_3d: np.ndarray [3]   (相机坐标 x,y,z meters)
  │       ├─ velocity_3d: np.ndarray [3]   (相机坐标 vx,vy,vz m/s)
  │       ├─ bbox_left:   [x1,y1,x2,y2]   (左相机像素坐标)
  │       ├─ bbox_right:  [x1,y1,x2,y2]   (右相机像素坐标)
  │       └─ target_id:   int
  │     imu_attitude: np.ndarray [3]       (roll, pitch, yaw rad)
  │     timestamp:    float                (seconds)
  │
  ▼
Pipeline.process_frame(frame_data)
  │
  ├─ Step1: StereoProcessor.rectify(left, right)
  │   └─→ rectified_left, rectified_right
  │
  ├─ Step2: Detector.detect(rectified_left, predicted_rois)
  │   └─→ List[Detection]
  │         ├─ bbox:       [x1,y1,x2,y2]
  │         ├─ confidence: float
  │         └─ class_id:   int
  │
  ├─ Step3: StereoProcessor.estimate_depth(detections, rect_L, rect_R)
  │   └─→ List[Detection3D]
  │         ├─ bbox:       [x1,y1,x2,y2]
  │         ├─ position_cam: np.ndarray [3]  (相机坐标系)
  │         ├─ depth:      float (meters)
  │         └─ confidence: float
  │
  ├─ Step4: Tracker.process_frame(detections_3d, timestamp)
  │   └─→ List[TrackState]
  │         ├─ track_id:    int
  │         ├─ state:       "tentative"/"confirmed"/"lost"
  │         ├─ position:    np.ndarray [3]
  │         ├─ velocity:    np.ndarray [3]
  │         ├─ acceleration: np.ndarray [3]
  │         ├─ covariance:  np.ndarray [9,9]
  │         └─ bbox:        [x1,y1,x2,y2]
  │
  └─ Step5: PositionSolver.solve(tracks, imu_attitude)
      └─→ List[TargetReport]
            ├─ track_id:      int
            ├─ azimuth_deg:   float  (方位角, 度)
            ├─ elevation_deg: float  (俯仰角, 度)
            ├─ slant_range_m: float  (斜距, 米)
            ├─ velocity_mps:  float  (相对速度, m/s)
            ├─ ttc_sec:       float  (碰撞预警时间, 秒)
            ├─ position_ned:  np.ndarray [3]  (NED坐标)
            └─ threat_level:  str    ("safe"/"warning"/"critical")
```

---

## 3. 坐标系定义

### 3.1 坐标系统一览

本系统涉及 5 个坐标系：

| 坐标系 | 符号 | 原点 | 轴定义 | 单位 |
|--------|------|------|--------|------|
| **像素坐标系** | (u, v) | 图像左上角 | u→右, v→下 | 像素 |
| **归一化相机坐标系** | (x_n, y_n) | 光心 | 同相机坐标系,z=1 | 无量纲 |
| **相机坐标系** | (X_c, Y_c, Z_c) | 左相机光心 | X→右, Y→下, Z→前 | 米 |
| **体坐标系** | (X_b, Y_b, Z_b) | 无人机质心 | X→前, Y→右, Z→下 | 米 |
| **NED坐标系** | (N, E, D) | 任意参考点 | N→北, E→东, D→下 | 米 |

### 3.2 坐标变换链

```
像素坐标 (u,v)
    │  K^(-1) 反投影
    ▼
归一化相机坐标 (x_n, y_n, 1)
    │  × depth Z
    ▼
相机坐标 (X_c, Y_c, Z_c)
    │  R_cam2body, T_cam2body
    ▼
体坐标 (X_b, Y_b, Z_b)
    │  R_body2ned(roll, pitch, yaw)
    ▼
NED坐标 (N, E, D)
```

### 3.3 关键变换矩阵

**内参矩阵 K**：

```
K = | fx   0  cx |
    |  0  fy  cy |
    |  0   0   1 |
```

**相机→体坐标变换**：

```
P_body = R_cam2body * P_cam + T_cam2body
```

其中 R_cam2body 由安装姿态（yaw, pitch, roll）决定。

**体→NED 坐标变换**：

```
P_ned = R_body2ned * P_body + P_self_ned
```

其中 R_body2ned 由 IMU 测量的（roll, pitch, yaw）构建。

---

## 4. 配置参数体系

### 4.1 配置文件结构

系统使用 YAML 格式配置文件 `config/default_config.yaml`，包含以下顶层分组：

| 配置组 | 前缀 | 内容 |
|--------|------|------|
| `camera` | `camera.*` | 相机内外参、畸变、安装偏移 |
| `detection` | `detection.*` | 检测器参数、目标物理尺寸 |
| `stereo_matching` | `stereo_matching.*` | SGBM参数、深度估计方法 |
| `tracking` | `tracking.*` | EKF参数、多目标管理参数 |
| `position_solver` | `position_solver.*` | 坐标系、TTC阈值 |
| `simulation` | `simulation.*` | 帧率、时长、随机种子、图像合成 |
| `output` | `output.*` | 输出路径、日志开关、可视化 |

### 4.2 配置加载方法

```python
import yaml

def load_config(config_path="config/default_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

---

## 5. 技术栈选型

| 层面 | 选型 | 版本要求 | 用途 |
|------|------|----------|------|
| 编程语言 | Python | ≥ 3.10 | 全部仿真代码 |
| 数值计算 | NumPy | ≥ 1.24 | 矩阵运算、数组操作 |
| 图像处理 | OpenCV | ≥ 4.8 | 立体视觉、图像合成 |
| 科学计算 | SciPy | ≥ 1.11 | 优化（Hungarian）、统计 |
| 可视化 | Matplotlib | ≥ 3.7 | 性能曲线、仿真报告图表 |
| 配置管理 | PyYAML | ≥ 6.0 | YAML 配置文件解析 |
| 测试框架 | unittest | 内置 | 单元测试和集成测试 |

---

## 6. 错误处理策略

| 异常类型 | 处理方式 |
|----------|----------|
| 配置文件缺失/格式错误 | 抛出 `FileNotFoundError` / `ValueError`，附带清晰提示 |
| 图像尺寸不匹配 | 校验后抛出 `ValueError` |
| 视差计算失败（无有效视差） | 返回 `depth=None`，标记置信度为 0 |
| 检测结果为空 | 正常处理，跟踪器仅做 predict 不做 update |
| 跟踪器状态爆炸（P矩阵过大） | 重新初始化该轨迹 |
| NaN/Inf 出现在数值计算中 | 检测并丢弃该帧结果，记录警告日志 |

---

## 7. 性能考量

### 7.1 仿真模式性能

在纯 CPU 仿真模式下，性能目标为：
- 单帧处理 < 500 ms（含图像合成）—— 仿真不要求实时
- 8 个场景 × 10 秒 × 120 fps = 9600 帧，总耗时 < 2 小时

### 7.2 部署模式性能参考

在 FPGA+GPU 异构平台上的目标延迟预算（参见系统方案文档第 4.2 节）。

### 7.3 内存预算

| 数据 | 单帧内存 | 说明 |
|------|----------|------|
| 左右图像 (uint8) | 2 × 6.2 MB = 12.4 MB | 1920×1080×3 |
| 视差图 (float32) | 8.3 MB | 1920×1080×1 |
| 跟踪器状态 (20轨迹) | < 0.1 MB | 9×9 协方差矩阵等 |
| 中间结果缓存 | < 5 MB | 检测框、深度值等 |
| **总计** | **< 30 MB** | 单帧峰值 |
