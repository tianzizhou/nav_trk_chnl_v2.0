# 目标检测模块 — 使用手册

## 1. 快速入门

```python
from src.detector import SimulatedDetector
import numpy as np

detector = SimulatedDetector()
detector.set_seed(42)

# 从 Ground Truth 生成模拟检测
# gt_list = frame.ground_truth (from SceneGenerator)
detections = detector.detect_from_gt(gt_list)

for det in detections:
    print(f"bbox={det.bbox}, conf={det.confidence:.2f}, "
          f"FA={det.is_false_alarm}")
```

---

## 2. 安装与依赖

仅需 NumPy（无 GPU 依赖）。

---

## 3. 配置参数详解

修改 `config/default_config.yaml` 中 `detection.simulated` 节：

| 参数 | 默认值 | 调节效果 |
|------|--------|----------|
| `bbox_noise_std_base` | 2.0 | 增大→检测框更不精确 |
| `detection_prob_base` | 0.98 | 减小→近距检测率降低 |
| `detection_prob_decay_range` | 1500.0 | 增大→远距检测率提高 |
| `false_alarm_rate` | 0.005 | 增大→更多虚警 |
| `confidence_threshold` | 0.3 | 增大→过滤更多低置信检测 |

---

## 4. API 参考

### 4.1 SimulatedDetector

```python
det = SimulatedDetector(config=None, rng=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | dict | 配置字典（None则加载默认） |
| `rng` | RandomState | 随机数生成器 |

### 4.2 detect_from_gt()

```python
detections = det.detect_from_gt(
    ground_truth_list,    # List[TargetGT]
    rois=None,            # Optional ROIs (unused)
    detector_overrides=None  # Optional param overrides
)
```

`detector_overrides` 示例：
```python
overrides = {'false_alarm_rate': 0.05}  # 10x normal FAR
dets = det.detect_from_gt(gt_list, detector_overrides=overrides)
```

### 4.3 set_seed()

```python
det.set_seed(42)  # For reproducibility
```

---

## 5. 使用示例

### 5.1 基本用法

```python
from src.detector import SimulatedDetector
from src.camera_model import StereoCamera
from src.scene_generator import SceneGenerator
from scenarios.s1_head_on import HeadOnScenario

camera = StereoCamera()
gen = SceneGenerator(camera)
det = SimulatedDetector()
det.set_seed(42)

scenario = HeadOnScenario(duration=5.0)
frame = gen.generate_frame(scenario, frame_idx=120)

detections = det.detect_from_gt(frame.ground_truth)
print(f"Detected {len(detections)} targets")
```

### 5.2 高虚警率测试

```python
# Stress test false alarm filtering
overrides = {'false_alarm_rate': 5.0}
dets = det.detect_from_gt(gt_list, detector_overrides=overrides)
real = [d for d in dets if not d.is_false_alarm]
fa = [d for d in dets if d.is_false_alarm]
print(f"Real: {len(real)}, FA: {len(fa)}")
```

### 5.3 实现自定义检测器

```python
from src.detector import DetectorBase, Detection

class MyDetector(DetectorBase):
    def detect(self, image, rois=None):
        # Your detection logic here
        results = my_model.predict(image)
        return [
            Detection(
                bbox=r['bbox'],
                confidence=r['score'],
                class_id=0,
            )
            for r in results
        ]
```

---

## 6. 输入/输出格式

### 输入：TargetGT

| 字段 | 要求 |
|------|------|
| `in_frame` | 必须为 True |
| `visible` | 必须为 True |
| `bbox_left` | 必须非 None，4 元素数组 |
| `distance` | 必须 > 0 |

### 输出：Detection

| 字段 | 类型 | 取值范围 |
|------|------|----------|
| `bbox` | ndarray[4] | 图像坐标范围内 |
| `confidence` | float | [conf_threshold, 1.0] |
| `class_id` | int | 0 |
| `is_false_alarm` | bool | True/False |

---

## 7. 常见问题 (FAQ)

**Q: 为什么远距离目标几乎没有检测？**

A: 模拟检测器使用 Sigmoid 概率模型，1500m 为半衰距离。
超过 2000m 检测概率 < 16%。可通过增大 `detection_prob_decay_range`
改善远距检测。

**Q: 如何完全禁用虚警？**

A: 设置 `false_alarm_rate: 0`。

**Q: 未来如何接入真实 YOLO 模型？**

A: 继承 `DetectorBase`，实现 `detect(image, rois)` 方法，
在 Pipeline 初始化时传入。详见设计文档第 7 节。

---

## 8. 故障排除

| 症状 | 原因 | 解决方法 |
|------|------|----------|
| 始终无检测 | 所有目标距离 > 2000m | 缩短初始距离或降低半衰距离 |
| 检测数波动大 | 目标在概率衰减区间 | 正常行为，多帧统计 |
| 虚警过多 | FAR 设置过高 | 降低 `false_alarm_rate` |
