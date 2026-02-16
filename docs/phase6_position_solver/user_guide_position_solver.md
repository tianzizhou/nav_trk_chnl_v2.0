# 位置解算模块 — 使用手册

## 快速使用

```python
from src.position_solver import PositionSolver

solver = PositionSolver()
# tracks = tracker.get_confirmed_tracks()
reports = solver.solve(tracks, imu_attitude=[0, 0, 0])

for r in reports:
    print(f"Target {r.track_id}: "
          f"Az={r.azimuth_deg:.1f}° "
          f"El={r.elevation_deg:.1f}° "
          f"Range={r.slant_range_m:.0f}m "
          f"TTC={r.ttc_sec:.1f}s "
          f"[{r.threat_level}]")
```

## 方位角约定

- 方位角：以正前方(Z轴)为0°，向右为正，范围[-180°, 180°]
- 俯仰角：水平为0°，向上为正（注意相机Y轴向下）

## 威胁等级

| 等级 | TTC阈值 | 建议动作 |
|------|---------|----------|
| safe | > 5s | 持续监视 |
| warning | 2~5s | 准备规避 |
| critical | < 2s | 立即规避 |
