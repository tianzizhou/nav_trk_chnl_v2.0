# 双目摄像头无人机检测仿真系统 — 系统使用总手册

## 1. 快速入门

### 1.1 五分钟上手

```bash
# Step 1: 安装依赖
pip install -r requirements.txt

# Step 2: 运行单场景仿真（正面迎头场景）
python sim/run_simulation.py --scenario s1_head_on

# Step 3: 运行全部场景批量评估
python sim/batch_evaluation.py --all

# Step 4: 生成仿真报告
python sim/report_generator.py --input results/ --output report/
```

### 1.2 环境要求

| 项目 | 要求 |
|------|------|
| Python | ≥ 3.10 |
| 操作系统 | Linux / macOS / Windows |
| 内存 | ≥ 4 GB |
| 磁盘空间 | ≥ 500 MB（含结果输出） |
| GPU | 不需要（仿真模式纯 CPU） |

---

## 2. 安装与依赖

### 2.1 依赖安装

```bash
pip install -r requirements.txt
```

依赖列表：

| 包名 | 版本 | 用途 |
|------|------|------|
| `opencv-python-headless` | ≥ 4.8.0 | 图像处理、立体视觉 |
| `scipy` | ≥ 1.11.0 | 匈牙利匹配、统计 |
| `matplotlib` | ≥ 3.7.0 | 可视化、报告图表 |
| `pyyaml` | ≥ 6.0 | 配置文件解析 |

### 2.2 验证安装

```bash
python -c "import cv2, scipy, matplotlib, yaml; print('All OK')"
```

---

## 3. 项目目录结构说明

```
/workspace/
├── README.md                   # 项目总览
├── requirements.txt            # Python 依赖
├── config/
│   └── default_config.yaml     # 默认配置文件
├── src/                        # 核心源码
│   ├── camera_model.py         # 双目相机模型
│   ├── utils.py                # 工具函数
│   ├── scene_generator.py      # 合成场景生成器
│   ├── stereo_processor.py     # 立体视觉处理
│   ├── detector.py             # 目标检测器
│   ├── tracker.py              # EKF 跟踪器
│   ├── position_solver.py      # 位置解算
│   └── pipeline.py             # 完整流水线
├── scenarios/                  # 场景定义
│   ├── scenario_base.py        # 场景基类
│   └── s1~s8_*.py              # 8 个仿真场景
├── tests/                      # 测试代码
├── sim/                        # 仿真运行脚本
│   ├── run_simulation.py       # 单场景运行
│   ├── batch_evaluation.py     # 批量评估
│   └── report_generator.py     # 报告生成
├── docs/                       # 文档（按阶段组织）
│   ├── system/                 # 系统级文档
│   ├── phase1~phase8_*/        # 各阶段文档
│   └── (35 篇 Markdown 文档)
└── results/                    # 仿真输出（运行时生成）
```

---

## 4. 配置文件详解

配置文件位于 `config/default_config.yaml`，采用 YAML 格式。

### 4.1 相机参数 (`camera`)

```yaml
camera:
  image_width: 1920          # 图像宽度（像素）
  image_height: 1080         # 图像高度（像素）
  left:
    fx: 1200.0               # X 方向焦距（像素）
    fy: 1200.0               # Y 方向焦距（像素）
    cx: 960.0                # 主点 X 坐标（像素）
    cy: 540.0                # 主点 Y 坐标（像素）
    distortion: [-0.1, 0.01, 0.0, 0.0, 0.0]  # [k1,k2,p1,p2,k3]
  stereo:
    baseline: 0.3            # 基线距离（米）
```

**参数调整指南**：
- 增大 `baseline` 可提高近距离深度精度，但远距离改善有限
- 增大 `fx/fy`（长焦距）可提高远距离检测灵敏度，但视场角缩小
- `distortion` 系数通常由相机标定得到

### 4.2 检测参数 (`detection`)

```yaml
detection:
  simulated:
    bbox_noise_std_base: 2.0      # 检测框噪声基准（像素）
    detection_prob_base: 0.98     # 100m 处检测概率
    detection_prob_decay_range: 1500.0  # 概率衰减距离（米）
    false_alarm_rate: 0.005       # 虚警率（每帧）
```

### 4.3 跟踪参数 (`tracking`)

```yaml
tracking:
  ekf:
    process_noise_accel_std: 50.0    # 过程噪声加速度标准差
  multi_target:
    association_gate: 50.0           # 关联门限（米）
    tentative_to_confirmed_hits: 3   # 确认所需连续命中数
```

### 4.4 仿真参数 (`simulation`)

```yaml
simulation:
  frame_rate: 120            # 帧率（Hz）
  default_duration: 10.0     # 默认仿真时长（秒）
  random_seed: 42            # 随机种子
  monte_carlo_runs: 100      # 蒙特卡洛运行次数
```

---

## 5. 运行单场景仿真

### 5.1 命令行方式

```bash
# 运行指定场景
python sim/run_simulation.py --scenario s1_head_on

# 指定自定义配置文件
python sim/run_simulation.py --scenario s2_high_speed_cross \
    --config config/custom_config.yaml

# 指定仿真时长和输出目录
python sim/run_simulation.py --scenario s3_tail_chase \
    --duration 15.0 --output results/s3/
```

### 5.2 Python 脚本方式

```python
from src.pipeline import DetectionPipeline
from scenarios.s1_head_on import HeadOnScenario

# 创建场景和流水线
scenario = HeadOnScenario()
pipeline = DetectionPipeline(config_path="config/default_config.yaml")

# 运行仿真
results = pipeline.run_sequence(scenario)

# 访问结果
for frame_result in results:
    for target in frame_result.target_reports:
        print(f"Target {target.track_id}: "
              f"Az={target.azimuth_deg:.1f}°, "
              f"Range={target.slant_range_m:.0f}m, "
              f"TTC={target.ttc_sec:.1f}s")
```

---

## 6. 运行批量评估

```bash
# 运行所有 8 个场景的评估
python sim/batch_evaluation.py --all

# 运行蒙特卡洛评估（100 次随机运行）
python sim/batch_evaluation.py --all --monte-carlo 100

# 仅运行指定场景
python sim/batch_evaluation.py --scenarios s1,s2,s5
```

---

## 7. 查看仿真报告

```bash
# 生成 Markdown 格式报告
python sim/report_generator.py --input results/ --output report/

# 报告内容包括：
# - 每个场景的性能指标汇总表
# - 检测率/虚警率统计
# - 3D 定位 RMSE 曲线
# - 跟踪 MOTA/MOTP 统计
# - 蒙特卡洛分布直方图
```

---

## 8. 常见问题 (FAQ)

### Q1: 运行时报 `ModuleNotFoundError`

确保从项目根目录运行，或设置 `PYTHONPATH`：

```bash
export PYTHONPATH=/workspace:$PYTHONPATH
```

### Q2: 如何添加自定义场景？

1. 在 `scenarios/` 目录下创建新文件，如 `s_custom.py`
2. 继承 `ScenarioBase` 基类
3. 实现 `get_ego_trajectory()` 和 `get_target_trajectories()` 方法
4. 在 `sim/run_simulation.py` 中注册新场景

### Q3: 如何替换为真实的 YOLO 检测器？

1. 继承 `DetectorBase` 抽象基类
2. 实现 `detect(image, rois=None)` 方法
3. 在 `Pipeline` 初始化时传入自定义检测器实例

### Q4: 仿真结果与预期不符怎么办？

1. 检查配置文件参数是否正确
2. 运行单元测试：`python -m pytest tests/ -v`
3. 查看各阶段的仿真文档中的预期值对照表
4. 启用逐帧日志：设置 `output.log_per_frame: true`

### Q5: 如何提高仿真速度？

- 减小仿真时长：`--duration 5.0`
- 降低蒙特卡洛次数：`--monte-carlo 10`
- 禁用图像合成（仅使用数值仿真）

---

## 9. 故障排除

| 症状 | 可能原因 | 解决方法 |
|------|----------|----------|
| `ImportError: cv2` | OpenCV 未安装 | `pip install opencv-python-headless` |
| 内存不足 | 图像缓存过多 | 减小仿真时长或降低分辨率 |
| 跟踪 ID 频繁跳变 | 关联门限过小 | 增大 `tracking.multi_target.association_gate` |
| 深度估计值为 None | 视差为 0 或无效 | 检查基线和焦距参数 |
| TTC 为负值 | 目标远离中 | 正常行为，表示目标在远离 |
