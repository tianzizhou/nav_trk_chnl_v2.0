# 目标检测模块 — 使用手册

## 快速使用

```python
from src.detector import SimulatedDetector

detector = SimulatedDetector()
detector.set_seed(42)

# gt_list = frame.ground_truth  (from SceneGenerator)
detections = detector.detect_from_gt(gt_list)
for det in detections:
    print(f"bbox={det.bbox}, conf={det.confidence:.2f}, "
          f"FA={det.is_false_alarm}")
```

## 参数调节

修改 `config/default_config.yaml` 中 `detection.simulated` 节。

## 接入真实YOLO

继承 `DetectorBase`，实现 `detect(image, rois)` 方法。
