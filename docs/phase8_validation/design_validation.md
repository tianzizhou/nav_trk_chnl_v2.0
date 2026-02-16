# 全场景仿真验证 — 设计文档

## 1. 仿真框架设计

```
sim/
├── run_simulation.py      # 单场景运行
│   ├── load_scenario()    # 场景注册表加载
│   ├── run_single()       # 运行+保存结果
│   └── save_results()     # JSON输出
│
├── batch_evaluation.py    # 批量评估
│   └── batch_evaluate()   # 遍历场景+汇总
│
└── report_generator.py    # 报告生成
    └── generate_report()  # Markdown报告
```

## 2. 场景注册表

```python
SCENARIOS = {
  's1_head_on': ('scenarios.s1_head_on', 'HeadOnScenario'),
  's2_high_speed_cross': (..., 'HighSpeedCrossScenario'),
  # ... s3~s8
}
```

## 3. 输出格式

### metrics.json

```json
{
  "scenario_name": "S1_Head_On",
  "num_frames": 100,
  "metrics": {
    "detection_rate": 0.85,
    "false_alarm_rate": 0.003,
    "position_rmse_m": 25.5
  }
}
```

### frame_data.jsonl

每行一帧，JSON Lines格式：

```json
{"frame_idx": 0, "timestamp": 0.0, "num_detections": 1,
 "num_confirmed_tracks": 0, "timing_ms": {...},
 "reports": [{...}]}
```

### batch_summary.json

所有场景指标汇总。

### evaluation_report.md

Markdown格式的人类可读报告。

## 4. 命令行接口

```bash
# 单场景
python sim/run_simulation.py -s s1_head_on -d 10 -f 120

# 批量
python sim/batch_evaluation.py --all -d 5 -f 30

# 报告
python sim/report_generator.py -i results/ -o report/
```
