# 目标跟踪模块 — 使用手册

## 1. 快速入门

```python
from src.tracker import MultiTargetTracker

tracker = MultiTargetTracker()

# Each frame: provide list of 3D detections
detections = [
    {'position': [10, -5, 200], 'bbox': [950, 530, 970, 540]},
    {'position': [-20, 3, 300], 'bbox': [900, 545, 930, 552]},
]
tracks = tracker.process_frame(detections, timestamp=0.0)

# Get confirmed targets only
for t in tracker.get_confirmed_tracks():
    print(f"Track {t.track_id}: "
          f"pos={t.position}, vel={t.velocity}, "
          f"status={t.status.value}")
```

---

## 2. 安装与依赖

```bash
pip install numpy scipy
```

---

## 3. 配置参数详解

### 3.1 EKF 参数

| 参数 | 默认值 | 说明 | 调节建议 |
|------|--------|------|----------|
| `process_noise_accel_std` | 50.0 | 加速度噪声 σ (m/s²) | 高机动目标增大，匀速目标减小 |

### 3.2 多目标管理参数

| 参数 | 默认值 | 说明 | 调节建议 |
|------|--------|------|----------|
| `association_gate` | 50.0 | 关联门限 (m) | 高速目标增大，密集目标减小 |
| `tentative_to_confirmed_hits` | 3 | 确认所需命中数 | 快速确认减小，减少虚警增大 |
| `confirmed_to_lost_misses` | 5 | 丢失所需缺失数 | 频繁漏检增大 |
| `lost_to_deleted_misses` | 10 | 删除所需缺失数 | 长遮挡增大 |
| `max_tracks` | 20 | 最大轨迹数 | 多目标场景增大 |

---

## 4. API 参考

### 4.1 MultiTargetTracker

```python
tracker = MultiTargetTracker(config=None)
```

#### process_frame(detections_3d, timestamp)

处理一帧检测结果，返回所有活跃轨迹。

| 参数 | 类型 | 说明 |
|------|------|------|
| `detections_3d` | List[dict] | 每个dict含'position'键 |
| `timestamp` | float | 当前时间戳 (可选) |
| **返回** | List[TrackState] | 所有非DELETED轨迹 |

检测字典格式：
```python
{'position': [x, y, z],   # 必需: 相机坐标系3D位置
 'bbox': [x1,y1,x2,y2],   # 可选: 像素检测框
 'confidence': 0.95}       # 可选: 检测置信度
```

#### get_confirmed_tracks()

仅返回 CONFIRMED 状态的轨迹。

#### get_predicted_rois(W, H, fx, fy, cx, cy, margin=100)

生成下一帧的预测搜索区域。

```python
rois = tracker.get_predicted_rois(1920, 1080, 1200, 1200, 960, 540)
# rois = [array([860, 440, 1060, 640]), ...]
```

#### reset()

清除所有轨迹，重置 ID 计数器。

### 4.2 EKFTracker（高级用法）

```python
from src.tracker import EKFTracker

ekf = EKFTracker(
    initial_state=[100, 0, 500],  # [x,y,z] or 9D
    process_noise_accel_std=30.0,
    dt=1/120,
)

ekf.predict()
ekf.update([101, 0.5, 498])

pos = ekf.get_position()    # [x, y, z]
vel = ekf.get_velocity()    # [vx, vy, vz]
acc = ekf.get_acceleration() # [ax, ay, az]

# Predict future position without modifying state
future_pos = ekf.get_predicted_position(dt=0.5)
```

### 4.3 TrackState 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `track_id` | int | 唯一轨迹 ID |
| `status` | TrackStatus | TENTATIVE/CONFIRMED/LOST/DELETED |
| `position` | ndarray[3] | 最新估计位置 (m) |
| `velocity` | ndarray[3] | 最新估计速度 (m/s) |
| `acceleration` | ndarray[3] | 最新估计加速度 (m/s²) |
| `covariance` | ndarray[9,9] | 状态协方差矩阵 |
| `bbox` | ndarray[4] | 最近匹配的检测框 |
| `hits` | int | 连续命中次数 |
| `misses` | int | 连续缺失次数 |
| `age` | int | 轨迹总帧龄 |

---

## 5. 使用示例

### 5.1 基本多目标跟踪

```python
from src.tracker import MultiTargetTracker

tracker = MultiTargetTracker()

# 模拟10帧数据
for frame in range(10):
    dets = [
        {'position': [100 + frame, 0, 500 - frame*10]},
        {'position': [-50, 20 + frame, 300]},
    ]
    tracks = tracker.process_frame(dets, timestamp=frame/120)

    confirmed = tracker.get_confirmed_tracks()
    print(f"Frame {frame}: {len(confirmed)} confirmed tracks")
    for t in confirmed:
        print(f"  ID={t.track_id}, pos={t.position}")
```

### 5.2 参数调优示例

```python
from src.utils import load_config

config = load_config()
# 高速目标场景：增大门限和过程噪声
config['tracking']['ekf']['process_noise_accel_std'] = 100.0
config['tracking']['multi_target']['association_gate'] = 100.0

tracker = MultiTargetTracker(config=config)
```

---

## 6. 常见问题 (FAQ)

**Q: 轨迹 ID 为什么跳变（如 1→3→5）？**

A: 中间的 ID 可能因为未确认就被删除了（TENTATIVE→DELETED）。
这是正常行为。

**Q: 如何降低 ID 切换率？**

A: (1) 减小 `association_gate` 避免交叉目标误关联；
(2) 未来增加外观特征匹配。

**Q: 远距离目标一直是 TENTATIVE 怎么办？**

A: 远距离检测率低导致连续命中不足。可减小
`tentative_to_confirmed_hits` 或提高检测率。

**Q: 轨迹位置出现突跳？**

A: 可能是过程噪声 Q 过小导致滤波器"信任"预测过度。
增大 `process_noise_accel_std`。

---

## 7. 故障排除

| 症状 | 原因 | 解决方法 |
|------|------|----------|
| 轨迹频繁创建删除 | 检测间断 | 增大 lost/deleted miss 阈值 |
| 两目标 ID 互换 | 交叉时关联错误 | 减小门限或增加外观特征 |
| 位置估计不收敛 | Q 或 R 设置不当 | 调整 accel_std 和确认参数 |
| 跟踪延迟太大 | 确认帧数太多 | 减小 tentative_to_confirmed_hits |
