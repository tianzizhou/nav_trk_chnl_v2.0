# 立体视觉处理 — 使用手册

## 1. 快速入门

```python
from src.camera_model import StereoCamera
from src.stereo_processor import StereoProcessor

camera = StereoCamera()
processor = StereoProcessor(camera)

# Estimate depth from known disparity
depth, conf = processor.estimate_depth_direct(
  disparity_value=3.6  # 3.6 pixels -> ~100m
)
print(f"Depth: {depth:.1f}m, Confidence: {conf:.2f}")

# Estimate depth from bbox height
depth, conf = processor.estimate_depth_direct(
  bbox_height=4.8  # 4.8 pixels -> ~100m
)
```

## 2. API 参考

### estimate_depth_direct(disparity_value, bbox_height)

简化的深度估计接口。

| 参数 | 类型 | 说明 |
|------|------|------|
| `disparity_value` | float | 视差值(像素), 可选 |
| `bbox_height` | float | 检测框高度(像素), 可选 |
| **返回** | (float, float) | (深度m, 置信度0~1) |

### estimate_depth(bbox, left_rect, right_rect)

完整深度估计（含SGBM计算）。

### compute_disparity_roi(left, right, bbox, padding=20)

对检测框区域计算SGBM视差。

## 3. 配置参数

修改 `config/default_config.yaml` 中 `stereo_matching` 节：

```yaml
stereo_matching:
  depth_method: "fusion"      # "disparity"/"size_prior"/"fusion"
  fusion_near_range: 100.0    # 纯视差区间上界
  fusion_far_range: 500.0     # 纯尺寸区间下界
```

## 4. FAQ

**Q: 如何提高远距离深度精度？**
A: 增大基线距离(baseline)或使用更长焦距。

**Q: SGBM计算太慢？**
A: 使用 `compute_disparity_roi()` 仅处理目标区域。
