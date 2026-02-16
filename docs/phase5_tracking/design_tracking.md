# 目标跟踪模块 — 详细设计文档

## 1. 模块架构

```
EKFTracker (单目标)              MultiTargetTracker
├─ x: ndarray[9]   (状态)       ├─ tracks: {id: EKFTracker}
├─ P: ndarray[9,9] (协方差)      ├─ track_meta: {id: TrackState}
├─ predict(dt)                   ├─ process_frame(dets, ts)
├─ update(measurement, R)       ├─ get_active_tracks()
├─ get_position()               ├─ get_confirmed_tracks()
├─ get_velocity()               ├─ get_predicted_rois()
├─ get_acceleration()           ├─ _associate(dets)
└─ get_predicted_position(dt)   ├─ _create_track(det)
                                └─ reset()
TrackStatus (Enum)
├─ TENTATIVE                    TrackState (dataclass)
├─ CONFIRMED                    ├─ track_id: int
├─ LOST                         ├─ status: TrackStatus
└─ DELETED                      ├─ position: ndarray[3]
                                ├─ velocity: ndarray[3]
                                ├─ acceleration: ndarray[3]
                                ├─ covariance: ndarray[9,9]
                                ├─ bbox: ndarray[4]
                                ├─ hits, misses, age: int
                                └─ last_timestamp: float
```

---

## 2. EKF 详细设计

### 2.1 状态向量（9维）

```
x = [x, y, z, vx, vy, vz, ax, ay, az]^T
```

- 位置 (x, y, z)：目标在相机坐标系中的 3D 位置（米）
- 速度 (vx, vy, vz)：相对速度（m/s）
- 加速度 (ax, ay, az)：相对加速度（m/s²）

### 2.2 状态转移矩阵 F（匀加速模型）

dt = 1/120 s = 0.00833 s

```
F = | I₃   dt·I₃   0.5·dt²·I₃ |
    | 0₃    I₃       dt·I₃     |
    | 0₃    0₃        I₃       |
```

展开为 9×9 矩阵：
```
F = | 1  0  0  dt  0   0   dt²/2  0      0     |
    | 0  1  0  0   dt  0   0      dt²/2  0     |
    | 0  0  1  0   0   dt  0      0      dt²/2 |
    | 0  0  0  1   0   0   dt     0      0     |
    | 0  0  0  0   1   0   0      dt     0     |
    | 0  0  0  0   0   1   0      0      dt    |
    | 0  0  0  0   0   0   1      0      0     |
    | 0  0  0  0   0   0   0      1      0     |
    | 0  0  0  0   0   0   0      0      1     |
```

### 2.3 过程噪声 Q（Singer 加速度模型）

假设加速度为一阶马尔可夫过程，过程噪声进入加速度分量。

每个轴的 3×3 噪声块（对应 [pos, vel, acc]）：

```
Q_1d = σ² × | dt⁵/20   dt⁴/8   dt³/6 |
              | dt⁴/8    dt³/3   dt²/2 |
              | dt³/6    dt²/2    dt   |
```

其中 σ = process_noise_accel_std = 50 m/s²。

完整 9×9 Q 矩阵为 3 个独立轴的 Q_1d 按对角排列。

### 2.4 测量模型 H

直接观测 3D 位置：

```
H = | 1  0  0  0  0  0  0  0  0 |
    | 0  1  0  0  0  0  0  0  0 |
    | 0  0  1  0  0  0  0  0  0 |
```

### 2.5 距离自适应测量噪声 R

测量噪声随目标距离增大（视差精度下降）：

```
σ_xy = max(2.0 × (Z/100), 0.5)     (米)
σ_z  = max(2.0 × (Z/100)², 0.5)    (米)
R = diag(σ_xy², σ_xy², σ_z²)
```

| 距离 Z (m) | σ_xy (m) | σ_z (m) | 说明 |
|------------|----------|---------|------|
| 50 | 1.0 | 0.5 | 近距高精度 |
| 100 | 2.0 | 2.0 | 基准 |
| 200 | 4.0 | 8.0 | 中距 |
| 500 | 10.0 | 50.0 | 远距噪声大 |
| 1000 | 20.0 | 200.0 | 深度噪声显著 |

### 2.6 初始协方差 P₀

```
P₀ = diag(100, 100, 100,        # 位置不确定性 (m²)
           2500, 2500, 2500,     # 速度不确定性 (m/s)²
           400, 400, 400)        # 加速度不确定性 (m/s²)²
```

---

## 3. 多目标关联（Hungarian 匹配）

### 3.1 代价矩阵构造

```
C[i,j] = ||pos_detection_i - pos_track_j||₂
```

如果 C[i,j] > gate_threshold (50m)，则设为 +∞（不允许匹配）。

### 3.2 匹配流程

```
1. 构建 N_det × N_trk 代价矩阵
2. scipy.optimize.linear_sum_assignment(cost)
3. 筛选：仅保留 cost < gate_threshold 的匹配
4. 分类：
   - 已匹配检测 → 更新对应轨迹
   - 未匹配检测 → 创建新轨迹
   - 未匹配轨迹 → 增加 miss 计数
```

---

## 4. 轨迹状态机

```
                ┌─────────────┐
                │  TENTATIVE  │
                │  (新建)      │
                └──────┬──────┘
                       │ hits ≥ 3
                       ▼
                ┌─────────────┐
        ┌──────│  CONFIRMED  │◄──────┐
        │      │  (已确认)    │       │
        │      └──────┬──────┘       │
        │             │ misses ≥ 5   │ hit (重新获取)
        │             ▼              │
        │      ┌─────────────┐       │
        │      │    LOST     │───────┘
        │      │  (丢失)     │
        │      └──────┬──────┘
        │             │ misses ≥ 10
        │             ▼
        │      ┌─────────────┐
        └─────→│   DELETED   │
               │  (删除)     │
               └─────────────┘

TENTATIVE: misses ≥ 10 也会直接 → DELETED
```

### 状态转换条件

| 转换 | 条件 |
|------|------|
| TENTATIVE → CONFIRMED | 连续命中 ≥ 3 帧 |
| CONFIRMED → LOST | 连续缺失 ≥ 5 帧 |
| LOST → CONFIRMED | 重新匹配到检测 |
| LOST → DELETED | 连续缺失 ≥ 10 帧 |
| TENTATIVE → DELETED | 连续缺失 ≥ 10 帧 |

---

## 5. 帧处理流程

```
process_frame(detections_3d, timestamp):
│
├─ Step 1: PREDICT all existing tracks
│   for each track: ekf.predict(dt)
│
├─ Step 2: ASSOCIATE detections ↔ tracks
│   cost_matrix = pairwise_distance(dets, tracks)
│   matched, unmatched_dets, unmatched_tracks
│     = hungarian_match(cost_matrix, gate=50m)
│
├─ Step 3: UPDATE matched tracks
│   for (det_idx, track_id) in matched:
│     track.ekf.update(det.position)
│     track.hits += 1; track.misses = 0
│     check status transition
│
├─ Step 4: HANDLE unmatched tracks (missed)
│   for track_id in unmatched_tracks:
│     track.misses += 1
│     check status transition (→ LOST / DELETED)
│
├─ Step 5: CREATE new tracks for unmatched detections
│   for det_idx in unmatched_dets:
│     new_track = EKFTracker(det.position)
│     status = TENTATIVE
│
├─ Step 6: REMOVE deleted tracks
│   delete all tracks with status == DELETED
│
└─ RETURN active track states
```

---

## 6. 配置参数

| 参数路径 | 默认值 | 说明 |
|----------|--------|------|
| `tracking.ekf.process_noise_accel_std` | 50.0 | σ_a (m/s²) |
| `tracking.multi_target.association_gate` | 50.0 | 门限 (m) |
| `tracking.multi_target.cost_weight_3d_distance` | 0.7 | 3D距离权重 |
| `tracking.multi_target.cost_weight_iou` | 0.3 | IoU权重(预留) |
| `tracking.multi_target.tentative_to_confirmed_hits` | 3 | 确认命中数 |
| `tracking.multi_target.confirmed_to_lost_misses` | 5 | 丢失缺失数 |
| `tracking.multi_target.lost_to_deleted_misses` | 10 | 删除缺失数 |
| `tracking.multi_target.max_tracks` | 20 | 最大轨迹数 |

---

## 7. 自适应 Q 的在线调整算法（预留设计）

当检测到滤波器创新序列异常大时，说明运动模型与实际不匹配，
需要临时增大过程噪声：

```
innovation = z_measured - H * x_predicted
innovation_cov = H * P * H^T + R
normalized_innovation = innovation^T * inv(S) * innovation

if normalized_innovation > chi2_threshold(0.99, df=3):
    # Target is maneuvering, increase Q
    Q_scale = max(normalized_innovation / 3, 1.0)
    Q_adapted = Q * Q_scale
```

此机制当前未激活，作为后续优化预留。
