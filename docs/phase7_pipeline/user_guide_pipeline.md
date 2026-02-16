# 处理流水线 — 使用手册

## 快速使用

```python
from src.pipeline import DetectionPipeline
from scenarios.s1_head_on import HeadOnScenario

pipeline = DetectionPipeline()
scenario = HeadOnScenario(duration=10.0)

result = pipeline.run_sequence(scenario, verbose=True)

print(f"Detection rate: {result.metrics['detection_rate']:.1%}")
print(f"Avg frame time: {result.metrics['avg_frame_time_ms']:.1f}ms")

for fr in result.frame_results:
    for report in fr.target_reports:
        print(f"t={fr.timestamp:.2f}s: "
              f"Az={report.azimuth_deg:.1f}° "
              f"Range={report.slant_range_m:.0f}m "
              f"TTC={report.ttc_sec:.1f}s "
              f"[{report.threat_level}]")
```

## API参考

- `DetectionPipeline(config_path="config/default_config.yaml")`
- `process_frame(frame_data)` → FrameResult
- `run_sequence(scenario, verbose=False)` → SequenceResult

## 结果数据

SequenceResult 包含：
- `frame_results`: 逐帧结果列表
- `metrics`: 评估指标字典
- `total_time_sec`: 总运行时间
