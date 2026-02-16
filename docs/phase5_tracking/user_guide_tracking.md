# 目标跟踪模块 — 使用手册

## 快速使用

```python
from src.tracker import MultiTargetTracker

tracker = MultiTargetTracker()

# Each frame: provide 3D detections
detections = [
  {'position': [10, -5, 200], 'bbox': [950, 530, 970, 540]},
]
tracks = tracker.process_frame(detections, timestamp=0.0)

for t in tracker.get_confirmed_tracks():
    print(f"Track {t.track_id}: pos={t.position}, "
          f"vel={t.velocity}")
```

## API参考

- `process_frame(dets, ts)` → List[TrackState]
- `get_confirmed_tracks()` → List[TrackState]
- `get_predicted_rois(W, H, fx, fy, cx, cy)` → List[ndarray]
- `reset()` → None

## 参数调优

- **高速目标**：增大 `association_gate`（默认50m）
- **多目标**：增大 `max_tracks`（默认20）
- **快速确认**：减小 `tentative_to_confirmed_hits`
