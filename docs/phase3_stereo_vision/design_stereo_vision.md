# 立体视觉处理 — 详细设计文档

## 1. 模块架构

```
StereoProcessor
├── compute_disparity_full()     # 全图SGBM视差
├── compute_disparity_roi()      # ROI局部SGBM视差
├── estimate_depth()             # 主入口：bbox + images → depth
├── estimate_depth_direct()      # 简化入口：disparity/bbox_h → depth
└── _fuse_depth()                # 融合视差法和尺寸法
```

## 2. 接口定义

```python
class StereoProcessor:
  def __init__(self, stereo_camera, config=None)
  def compute_disparity_full(self, left_rect, right_rect)
      -> np.ndarray  # [H,W] float32
  def compute_disparity_roi(self, left, right, bbox, padding=20)
      -> float  # median disparity
  def estimate_depth(self, bbox, left=None, right=None,
                     disparity_value=None)
      -> (float, float)  # (depth_m, confidence)
  def estimate_depth_direct(self, disparity_value=None,
                            bbox_height=None)
      -> (float, float)
```

## 3. 核心算法

### 3.1 SGBM 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| numDisparities | 256 | 视差搜索范围 (必须被16整除) |
| blockSize | 5 | 匹配块大小 |
| P1 | 200 | 视差平滑惩罚（相邻变化1） |
| P2 | 800 | 视差平滑惩罚（相邻变化>1） |
| uniquenessRatio | 10 | 最佳匹配唯一性比率(%) |

### 3.2 融合权重

```
if depth <= fusion_near (100m):
    w_disp = 1.0, w_size = 0.0
elif depth >= fusion_far (500m):
    w_disp = 0.0, w_size = 1.0
else:
    w_disp = (fusion_far - depth) / (fusion_far - fusion_near)
    w_size = 1.0 - w_disp

fused = w_disp * depth_disp + w_size * depth_size
```

## 4. 配置参数

| 路径 | 默认值 | 说明 |
|------|--------|------|
| `stereo_matching.depth_method` | "fusion" | 方法选择 |
| `stereo_matching.fusion_near_range` | 100.0 | 纯视差范围上界(m) |
| `stereo_matching.fusion_far_range` | 500.0 | 纯尺寸范围下界(m) |
| `stereo_matching.sgbm.*` | 见配置文件 | SGBM参数组 |
