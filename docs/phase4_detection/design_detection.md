# 目标检测模块 — 详细设计文档

## 1. 模块架构

```
DetectorBase (ABC)              Detection (dataclass)
├─ detect(image, rois)          ├─ bbox: ndarray[4]  [x1,y1,x2,y2]
│  -> List[Detection]           ├─ confidence: float [0,1]
│                               ├─ class_id: int     (0=drone)
SimulatedDetector               └─ is_false_alarm: bool
├─ detect_from_gt(gt_list,
│    rois, overrides)
├─ _detection_probability(dist)
├─ _confidence_score(dist)
├─ _generate_false_alarm()
└─ set_seed(seed)
```

---

## 2. 类/函数接口定义

### 2.1 DetectorBase 抽象基类

```python
class DetectorBase(ABC):
  @abstractmethod
  def detect(self, image: np.ndarray,
             rois: Optional[List] = None
             ) -> List[Detection]:
    """Detect targets in image.
    Args:
      image: [H, W, 3] uint8 BGR image
      rois:  Optional [[x1,y1,x2,y2], ...] search regions
    Returns:
      List of Detection objects
    """
    pass
```

### 2.2 SimulatedDetector 类

```python
class SimulatedDetector(DetectorBase):
  def __init__(self, config=None, rng=None)
  def detect(self, image, rois=None) -> List[Detection]
  def detect_from_gt(self, ground_truth_list,
                     rois=None,
                     detector_overrides=None
                     ) -> List[Detection]
  def set_seed(self, seed: int) -> None
```

### 2.3 Detection 数据类

| 字段 | 类型 | 说明 |
|------|------|------|
| `bbox` | ndarray[4] | [x1, y1, x2, y2] 像素坐标 |
| `confidence` | float | 检测置信度 [0, 1] |
| `class_id` | int | 类别ID（0=无人机） |
| `is_false_alarm` | bool | 是否为虚警（评估用） |

---

## 3. 核心算法

### 3.1 检测概率模型（Sigmoid 衰减）

```python
def _detection_probability(self, distance):
    scale = decay_range / 5.0
    prob = base_prob / (1 + exp((distance - decay_range) / scale))
    return clip(prob, 0, 1)
```

参数：
- `base_prob` = 0.98
- `decay_range` = 1500.0m
- `scale` = 300.0m

### 3.2 bbox 噪声模型（距离缩放高斯）

```python
noise_scale = distance / 100.0
bbox_noise = randn(4) * bbox_noise_std_base * noise_scale
noisy_bbox = gt_bbox + bbox_noise
```

- 基准噪声 σ = 2.0 像素 @ 100m
- 500m 处：σ = 10.0 像素
- 1000m 处：σ = 20.0 像素

### 3.3 置信度模型（线性衰减 + 随机扰动）

```python
base_conf = clip(1.0 - distance / 3000.0, 0.2, 0.99)
confidence = clip(base_conf + randn() * 0.05, 0, 1)
```

### 3.4 虚警生成模型（泊松分布）

```python
num_fa = Poisson(false_alarm_rate)
for each FA:
    bbox = random_bbox(w=U(5,30), h=U(3,15),
                       pos=U(0,W) × U(0,H))
    conf = U(conf_threshold, 0.5)
```

### 3.5 ROI 预测检测流程

当跟踪器提供预测 ROI 时：

```
1. 获取预测位置 → 投影到图像平面
2. 搜索窗口 = 预测中心 ± 3σ_position
3. 仅在 ROI 内运行检测
4. 每 N 帧(默认4帧)执行一次全图检测发现新目标
```

---

## 4. 配置参数

| 参数路径 | 默认值 | 范围 | 说明 |
|----------|--------|------|------|
| `detection.simulated.bbox_noise_std_base` | 2.0 | 0~10 | 100m处bbox噪声标准差(px) |
| `detection.simulated.detection_prob_base` | 0.98 | 0~1 | 100m处检测概率 |
| `detection.simulated.detection_prob_decay_range` | 1500.0 | 100~5000 | 概率半衰距离(m) |
| `detection.simulated.false_alarm_rate` | 0.005 | 0~1 | 每帧虚警率(泊松λ) |
| `detection.simulated.confidence_threshold` | 0.3 | 0~1 | 最低置信度过滤 |
| `detection.full_detect_interval` | 4 | 1~10 | 全图检测间隔(帧) |
| `detection.target_size.wingspan` | 1.2 | - | 目标翼展(m) |
| `detection.target_size.height` | 0.4 | - | 目标高度(m) |

---

## 5. 错误处理

| 条件 | 处理方式 |
|------|----------|
| GT 目标不在视场内 | 跳过，不生成检测 |
| GT 目标被遮挡 | 跳过（visible=False） |
| bbox 超出图像范围 | 裁剪到图像边界 |
| bbox 无效（x2≤x1 或 y2≤y1） | 丢弃该检测 |
| 置信度低于阈值 | 丢弃该检测 |

---

## 6. 性能考量

- 模拟检测器延迟 < 0.1ms（纯数学计算）
- 真实 YOLOv8n TensorRT 延迟 ~3ms (Jetson Orin)
- 虚警生成为独立的泊松过程，不依赖图像内容

---

## 7. 未来扩展：接入真实 YOLO 模型

```python
class YOLODetector(DetectorBase):
    def __init__(self, model_path, device='cuda'):
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect(self, image, rois=None):
        results = self.model(image)
        detections = []
        for r in results[0].boxes:
            detections.append(Detection(
                bbox=r.xyxy[0].cpu().numpy(),
                confidence=float(r.conf[0]),
                class_id=int(r.cls[0]),
            ))
        return detections
```

替换步骤：
1. 安装 `ultralytics` 包
2. 准备无人机检测模型权重文件
3. 在 Pipeline 初始化时传入 `YOLODetector` 实例
