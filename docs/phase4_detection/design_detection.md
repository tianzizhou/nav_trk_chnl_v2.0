# 目标检测模块 — 详细设计文档

## 1. 类图

```
DetectorBase (ABC)         Detection (dataclass)
├─ detect()                ├─ bbox [4]
                           ├─ confidence float
SimulatedDetector          ├─ class_id int
├─ detect_from_gt()        └─ is_false_alarm bool
├─ _detection_probability()
├─ _confidence_score()
├─ _generate_false_alarm()
└─ set_seed()
```

## 2. 接口定义

```python
class DetectorBase(ABC):
  def detect(self, image, rois=None) -> List[Detection]

class SimulatedDetector(DetectorBase):
  def __init__(self, config=None, rng=None)
  def detect_from_gt(self, ground_truth_list, rois=None,
                     detector_overrides=None) -> List[Detection]
  def set_seed(self, seed)
```

## 3. 检测概率模型

```
P(d) = base_prob / (1 + exp((d - decay_range) / scale))
scale = decay_range / 5
```

## 4. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bbox_noise_std_base` | 2.0 | 100m处bbox噪声(px) |
| `detection_prob_base` | 0.98 | 100m处检测概率 |
| `detection_prob_decay_range` | 1500.0 | 半衰距离(m) |
| `false_alarm_rate` | 0.005 | 每帧虚警率 |
| `confidence_threshold` | 0.3 | 最低置信度阈值 |
