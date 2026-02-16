# 全场景仿真验证 — 使用手册

## 1. 运行单场景仿真

```bash
# 基本用法
python sim/run_simulation.py --scenario s1_head_on

# 自定义参数
python sim/run_simulation.py \
  --scenario s2_high_speed_cross \
  --duration 5.0 \
  --frame-rate 30 \
  --output results/custom/

# 安静模式
python sim/run_simulation.py -s s1_head_on -q
```

### 可用场景

| 场景名 | 说明 |
|--------|------|
| `s1_head_on` | S1: 正面迎头接近 |
| `s2_high_speed_cross` | S2: 高速交叉飞越 |
| `s3_tail_chase` | S3: 尾追同向飞行 |
| `s4_multi_target` | S4: 多目标群攻 |
| `s5_maneuvering` | S5: 目标机动规避 |
| `s6_long_range` | S6: 远距离检测 |
| `s7_clutter` | S7: 杂波干扰 |
| `s8_occlusion` | S8: 多目标遮挡 |

## 2. 运行批量评估

```bash
# 运行全部8个场景
python sim/batch_evaluation.py --all

# 运行指定场景
python sim/batch_evaluation.py --scenarios s1_head_on s2_high_speed_cross

# 快速验证模式（低帧率）
python sim/batch_evaluation.py --all -d 3.0 -f 10
```

## 3. 生成仿真报告

```bash
python sim/report_generator.py --input results/ --output report/
```

报告输出为 `report/evaluation_report.md`，包含：
- 场景指标汇总表
- 每个场景的详细结果
- 结论与建议

## 4. 输出文件说明

```
results/
├── batch_summary.json          # 批量评估汇总
├── s1_head_on/
│   ├── metrics.json            # 场景指标
│   └── frame_data.jsonl        # 逐帧数据
├── s2_high_speed_cross/
│   └── ...
└── ...

report/
└── evaluation_report.md        # Markdown报告
```

## 5. 常见问题

**Q: 仿真太慢怎么办？**

A: 使用低帧率进行快速验证：
```bash
python sim/batch_evaluation.py --all -d 3.0 -f 10
```

**Q: 如何添加新场景到评估？**

1. 在 `scenarios/` 创建场景文件
2. 在 `sim/run_simulation.py` 的 `SCENARIOS` 字典中注册
3. 重新运行 batch_evaluation

**Q: 结果中检测率很低正常吗？**

A: 短时仿真中目标多在远距离（>1500m），检测概率自然偏低。
延长仿真时长（如10s）可看到随距离缩短检测率上升的趋势。
