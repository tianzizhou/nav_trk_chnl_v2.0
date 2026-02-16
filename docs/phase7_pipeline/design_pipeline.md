# 处理流水线 — 详细设计文档

## 1. DetectionPipeline 类

```python
class DetectionPipeline:
  def __init__(self, config=None, config_path=None)
  def process_frame(self, frame_data, frame_count=0)
      -> FrameResult
  def run_sequence(self, scenario, verbose=False)
      -> SequenceResult
```

## 2. 数据流

```
FrameData
├─ left_image, right_image
├─ ground_truth: List[TargetGT]
└─ imu_attitude: [roll, pitch, yaw]
        │
        ▼
SimulatedDetector.detect_from_gt()
        │
        ▼ List[Detection]
        │
StereoProcessor.estimate_depth_direct()
        │
        ▼ List[dict] (3D detections)
        │
MultiTargetTracker.process_frame()
        │
        ▼ List[TrackState]
        │
PositionSolver.solve()
        │
        ▼ List[TargetReport]
```

## 3. FrameResult 数据结构

| 字段 | 类型 | 说明 |
|------|------|------|
| frame_idx | int | 帧编号 |
| timestamp | float | 时间戳 |
| detections | list | 检测结果 |
| tracks | list | 所有活跃轨迹 |
| target_reports | List[TargetReport] | 已确认目标报告 |
| ground_truth | list | GT参考 |
| timing_ms | dict | 各模块耗时 |

## 4. 评估指标计算

- detection_rate = 真实检测数 / 可见GT数
- false_alarm_rate = FA数 / 总帧数
- position_rmse_m = sqrt(mean(||pos_est - pos_gt||²))
- range_rel_error_mean = mean(|range_est - range_gt| / range_gt)
- azimuth_error_deg_mean = mean(|az_est - az_gt|)
