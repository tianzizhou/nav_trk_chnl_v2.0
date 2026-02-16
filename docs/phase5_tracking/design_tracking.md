# 目标跟踪模块 — 详细设计文档

## 1. EKFTracker 设计

### 状态向量 (9D)

x = [x, y, z, vx, vy, vz, ax, ay, az]^T

### 状态转移矩阵

```
F = | I   dt*I   0.5*dt²*I |
    | 0    I       dt*I    |
    | 0    0        I      |
```

### 过程噪声 (Singer模型)

每轴Q矩阵（3x3块）：
```
Q_1d = σ² * | dt⁵/20  dt⁴/8  dt³/6 |
             | dt⁴/8   dt³/3  dt²/2 |
             | dt³/6   dt²/2   dt   |
```

### 测量模型

H = [I₃ | 0₃ | 0₃]，直接观测3D位置

### 距离自适应测量噪声

R = diag(σ_xy², σ_xy², σ_z²)
σ_z = 2.0 * (Z/100)²，σ_xy = 2.0 * (Z/100)

## 2. MultiTargetTracker 设计

### 轨迹状态机

```
TENTATIVE ──(hits≥3)──> CONFIRMED
CONFIRMED ──(miss≥5)──> LOST
LOST ──(hit)──────────> CONFIRMED
LOST ──(miss≥10)──────> DELETED
```

### 关联代价矩阵

C_ij = ||pos_det_i - pos_trk_j||₂

门限：50m（超过不关联）

## 3. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `process_noise_accel_std` | 50.0 | 加速度噪声 (m/s²) |
| `association_gate` | 50.0 | 关联门限 (m) |
| `tentative_to_confirmed_hits` | 3 | 确认所需命中次数 |
| `confirmed_to_lost_misses` | 5 | 丢失所需缺失次数 |
| `lost_to_deleted_misses` | 10 | 删除所需缺失次数 |
