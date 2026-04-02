# 场景自适应融合策略

## 1 文档信息

| 项目 | 内容 |
|------|------|
| 文档编号 | CUAV-SCN-005 |
| 版本 | V1.0.0 |
| 日期 | 2026-02-20 |

---

## 2 概述

本文档详细描述多模态融合系统在不同作战场景下的自适应融合策略，
包括场景分类方法、传感器可信度动态评估、各场景专项融合流程以及
传感器协同管理机制。

系统面临的核心挑战是：不同场景下各传感器的有效性差异巨大，必须
根据环境条件动态调整融合策略，才能保证全场景下的探测性能。

---

## 3 场景分类体系

### 3.1 场景维度定义

系统从以下维度对当前场景进行分类：

| 维度 | 参数 | 来源 | 分级 |
|------|------|------|------|
| 目标高度 | h (m) | 雷达/光电测量 | 超低空(<10m)/低空(10-100m)/中空(>100m) |
| 目标距离 | R (m) | 雷达/融合估计 | 近距(<200m)/中距(200-1000m)/远距(>1000m) |
| 杂波强度 | CNR (dB) | 雷达杂波图 | 强杂波(>20dB)/中杂波/弱杂波(<5dB) |
| 能见度 | vis (m) | 气象传感器/手动 | 良好(>5km)/一般(1-5km)/恶劣(<1km) |
| 环境噪声 | SNR_ac (dB) | 声学阵列 | 安静(>20dB)/一般(10-20dB)/嘈杂(<10dB) |
| 地形类型 | type | 手动/地图 | 开阔/郊区/城市/山地 |
| 目标运动 | pattern | IMM模型概率 | 匀速/转弯/悬停/蛙跳 |
| RF环境 | rf_det | RF侦测 | 有信号/无信号 |

### 3.2 复合场景定义

将多个维度组合为典型作战场景：

| 场景编号 | 场景名称 | 关键特征 |
|---------|---------|---------|
| S1 | 开阔-中远距-正常飞行 | h>10m, R>500m, 弱杂波, 良好能见度 |
| S2 | 超低空飞行 | h<10m, 强地杂波 |
| S3 | 近距快速接近 | R<200m, 高角速度 |
| S4 | 蛙跳机动 | 飞行-悬停交替 |
| S5 | 城市复杂背景 | 城市地形, 遮挡/多径 |
| S6 | 恶劣天气 | 能见度<1km, 降雨 |
| S7 | 自主飞行(无RF) | 无RF信号检测 |
| S8 | 多目标密集 | 同时>5批目标 |

### 3.3 场景分类器

场景分类器实时运行，基于规则和传感器反馈判断当前场景：

```
function classify_scene(sensor_status, env_params, track_info):
    scene_vector = []

    // 高度维度
    if track_info.min_altitude < 10:
        scene_vector.append(S2)
    elif track_info.max_altitude > 100:
        scene_vector.append(S1_HIGH)

    // 距离维度
    if track_info.min_range < 200:
        scene_vector.append(S3)

    // 杂波维度
    if sensor_status.radar_cnr > 20:
        scene_vector.append(S2)

    // 天气维度
    if env_params.visibility < 1000:
        scene_vector.append(S6)

    // 运动模式维度
    if track_info.frog_leap_detected:
        scene_vector.append(S4)

    // 地形维度
    if env_params.terrain_type == URBAN:
        scene_vector.append(S5)

    // RF维度
    if not sensor_status.rf_signal_detected:
        scene_vector.append(S7)

    // 目标数维度
    if track_info.confirmed_track_count > 5:
        scene_vector.append(S8)

    // 如果无特殊场景, 默认 S1
    if len(scene_vector) == 0:
        scene_vector = [S1]

    return scene_vector
```

**注意**：多个场景可以同时有效（如超低空 + 城市），
此时取各场景权重的加权组合。

---

## 4 传感器可信度评估

### 4.1 先验可信度矩阵

基于各场景下传感器的物理特性，预设先验可信度权重：

```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┐
│ 场景             │ W_radar  │ W_eo     │ W_acou   │ W_rf     │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ S1: 开阔/正常    │   0.40   │   0.35   │   0.10   │   0.15   │
│ S2: 超低空       │   0.10   │   0.35   │   0.35   │   0.20   │
│ S3: 近距         │   0.25   │   0.15   │   0.40   │   0.20   │
│ S4: 蛙跳         │   0.15   │   0.30   │   0.35   │   0.20   │
│ S5: 城市         │   0.15   │   0.25   │   0.30   │   0.30   │
│ S6: 恶劣天气     │   0.35   │   0.10   │   0.30   │   0.25   │
│ S7: 无RF         │   0.35   │   0.35   │   0.30   │   0.00   │
│ S8: 多目标       │   0.40   │   0.30   │   0.15   │   0.15   │
└──────────────────┴──────────┴──────────┴──────────┴──────────┘
```

### 4.2 多场景叠加

当多个场景同时生效时，取加权平均：

\[
w_s^{scene} = \frac{\sum_{k} \alpha_k \cdot w_s^{S_k}}{\sum_{k} \alpha_k}
\]

其中 \(\alpha_k\) 为场景 \(S_k\) 的激活强度（0~1），表示该场景
的严重程度。

例如：目标在城市超低空飞行
- S2（超低空）激活强度 0.8
- S5（城市）激活强度 0.7

\[
w_{radar}^{scene} = \frac{0.8 \times 0.10 + 0.7 \times 0.15}{0.8 + 0.7} = \frac{0.185}{1.5} = 0.123
\]

### 4.3 基于残差的实时可信度调整

先验权重仅反映"预期"可信度，实际运行中需要根据传感器的表现
动态调整。核心指标是 **归一化新息残差**：

\[
\epsilon_s(k) = \frac{1}{n_z} \sum_{l=1}^{n_z} \frac{v_{s,l}^2(k)}{S_{s,ll}(k)}
\]

其中 \(v_s(k)\) 为传感器 s 的新息向量，\(S_s(k)\) 为新息协方差。

理想情况下 \(\epsilon_s \approx 1\)（新息白化残差为单位方差）。

**自适应权重更新**：

\[
w_s(k) = \frac{w_s^{scene} \cdot \exp(-\lambda \cdot (\epsilon_s(k) - 1)^2)}{\sum_{s'} w_{s'}^{scene} \cdot \exp(-\lambda \cdot (\epsilon_{s'}(k) - 1)^2)}
\]

参数 \(\lambda = 2.0\) 控制调整灵敏度。

**物理含义**：
- 当传感器新息残差接近理想值 1 时，权重基本不变
- 当残差偏大（量测不可靠）时，权重降低
- 当残差偏小（量测噪声被高估）时，权重适当提高

### 4.4 权重 → 量测噪声映射

传感器可信度权重通过调整量测噪声矩阵来影响融合：

\[
\mathbf{R}_s^{adaptive}(k) = \frac{\mathbf{R}_s^{nominal}}{w_s(k)}
\]

效果：
- 高权重（可信度高）→ 噪声小 → 融合中贡献大
- 低权重（可信度低）→ 噪声大 → 融合中贡献小
- 权重为 0 → 噪声无穷大 → 等效不更新（传感器排除）

---

## 5 各场景详细融合策略

### 5.1 场景 S1：开阔地域-中远距-正常飞行

**场景特征**：
- 目标高度 > 10 m
- 距离 > 500 m
- 地形开阔，杂波水平正常
- 良好能见度

**传感器角色**：
- 雷达：主传感器（距离/速度）
- 光电：辅助传感器（高精度角度/识别）
- 声学：在范围内时辅助确认
- RF：被动探测辅助

**融合流程**：

```
┌────────────────────────────────────────────────────────────┐
│ S1: 标准融合流程                                           │
│                                                            │
│  [雷达搜索] ──→ CFAR检测 ──→ 点迹 ──→ M/N航迹起始          │
│       │                                   │                │
│       │                              暂态航迹               │
│       │                                   │                │
│  [光电] ←──── 雷达引导光电指向 ──────────┘                 │
│       │                                                    │
│       │    光电捕获 ──→ 确认航迹                           │
│       │         │                                          │
│       │    IMM-UKF 标准序贯更新:                           │
│       │    雷达(R,Az,El,Vr) → 光电(Az,El)                 │
│       │         │                                          │
│       │    融合航迹 ──→ 指控系统                           │
│       │                                                    │
│  特殊处理:                                                 │
│  · 声学在300m内加入序贯更新                                │
│  · RF信号检测作为分类辅助                                  │
│  · IMM以CV模型为主(概率~0.9)                               │
└────────────────────────────────────────────────────────────┘
```

**关键参数**：
- 融合周期：200 ms
- 关联门限：标准（γ 按卡方分布）
- 航迹确认：雷达 2/3 + 光电确认
- 外推时间：4 s

### 5.2 场景 S2：超低空飞行（h < 10 m）

**场景特征**：
- 目标高度 < 10 m
- 雷达强地杂波（CNR > 20 dB），检测概率显著下降
- 目标可能利用地形遮蔽

**核心挑战**：雷达几乎失效

**传感器角色**：
- 雷达：辅助（杂波图增强模式，检测概率低但不放弃）
- 光电：主传感器之一（不受杂波影响，但需要引导）
- 声学：主传感器（300m内可靠探测，不受杂波影响）
- RF：重要辅助（被动探测不受影响）

**融合流程**：

```
┌────────────────────────────────────────────────────────────┐
│ S2: 超低空融合流程                                         │
│                                                            │
│  [声学阵列] ──→ MUSIC DOA ──→ 声学暂态航迹                │
│                                     │                      │
│  [RF侦测]  ──→ 测向 ──→ RF暂态航迹                        │
│                              │      │                      │
│                              │      ▼                      │
│                    角度域多源关联                           │
│                         (马氏距离, 仅角度维)               │
│                              │                             │
│                              ▼                             │
│                    声学+RF确认 → 确认航迹                  │
│                              │                             │
│  [光电] ←──── DOA引导 ──────┘                             │
│       │                                                    │
│       │   光电捕获 ──→ 角度精化                            │
│       │        │                                           │
│  [雷达] ──→ 杂波图增强模式                                │
│       │        │                                           │
│       │   雷达检测到目标?                                  │
│       │     ├─ 是 → 完整序贯更新(含距离)                  │
│       │     └─ 否 → 纯角度跟踪 + 高度约束测距             │
│       │                                                    │
│  纯角度测距:                                               │
│    R_est = (h_assumed - h_platform) / sin(El)              │
│    h_assumed = 5m (先验), sigma_h = 3m                     │
│    作为伪量测加入UKF更新                                   │
│                                                            │
│  特殊处理:                                                 │
│  · 雷达CNR > 25dB时: W_radar → 0.05                       │
│  · 声学+光电角度交叉定位辅助测距                           │
│  · 航迹外推时间延长至6s(低空目标速度慢)                    │
└────────────────────────────────────────────────────────────┘
```

**角度交叉定位**：

当仅有光电和声学两个角度传感器时，若已知目标大致高度，
可用三角测量辅助估计距离：

\[
R \approx \frac{h_{target}}{\sin(El_{avg})}
\]

其中 \(El_{avg}\) 为光电和声学俯仰角的加权平均。

**距离估计不确定性传播**：

\[
\sigma_R = \frac{h_{target}}{\sin^2(El)} \cdot \sigma_{El} + \frac{\sigma_h}{\sin(El)}
\]

典型值（h=5m, El=3°, σ_El=0.5mrad, σ_h=3m）：

\[
\sigma_R \approx 57 \text{ m} + 57 \text{ m} = 114 \text{ m}
\]

虽然距离精度较差，但可以提供粗略的距离估计。

### 5.3 场景 S3：近距快速接近（R < 200 m）

**场景特征**：
- 目标距离 < 200 m
- 目标角速度很大（可达 > 30°/s）
- 光电窄视场无法跟踪

**核心挑战**：光电窄视场脱跟

**传感器角色**：
- 雷达：可靠（近距SNR高，测距好）
- 光电-窄视场：脱跟（切换到广角）
- 光电-广角：接替跟踪（精度降低但不脱跟）
- 声学：主角度传感器（大视场，不受角速度影响）
- RF：辅助

**融合流程**：

```
┌────────────────────────────────────────────────────────────┐
│ S3: 近距融合流程                                           │
│                                                            │
│  触发条件: R < 200m 或 目标角速度 > 30°/s                  │
│                                                            │
│  [光电系统]                                                │
│  ┌────────────────────────────────────────────────┐        │
│  │ 窄视场 ──→ 角速度监测 ──→ θ_dot > 30°/s?      │        │
│  │   │                          ├─ 否: 继续窄视场 │        │
│  │   │                          └─ 是: ↓          │        │
│  │   │                     切换广角相机            │        │
│  │   │                          │                  │        │
│  │   │            广角YOLO检测 + KCF跟踪           │        │
│  │   │                          │                  │        │
│  │   │            输出角度(精度: ~0.5°)             │        │
│  └───│──────────────────────────│──────────────────┘        │
│      │                          │                           │
│  [声学]  ──→ 高刷新率MUSIC(20Hz) ──→ 角度(精度: ~3°)       │
│      │                          │                           │
│  [雷达]  ──→ 近距模式 ──→ 距离+速度(高SNR,精度好)           │
│      │                          │                           │
│      └──────────┬───────────────┘                           │
│                 ▼                                           │
│         序贯融合(高更新率模式):                             │
│         雷达距离(2m精度) + 广角角度(0.5°) + 声学角度(3°)    │
│                 │                                           │
│         状态更新率: 50 Hz (打击引导需求)                    │
│                 │                                           │
│         输出 → 打击分系统                                   │
│                                                            │
│  特殊处理:                                                 │
│  · 融合周期从200ms缩短至20ms                               │
│  · 声学刷新率提升至20Hz                                    │
│  · 雷达工作在近距高分辨模式                                │
│  · 广角相机帧率提升至30fps                                 │
│  · IMM以CT模型为主(近距目标常转弯规避)                     │
│  · 关联门限适当放宽(目标运动不确定性大)                    │
└────────────────────────────────────────────────────────────┘
```

**广角/窄视场切换逻辑详细设计**：

```
function eo_fov_switch(track, eo_status):
    // 计算目标角速度
    theta_dot = compute_angular_rate(track)

    // 切换状态机
    switch eo_status.fov_mode:
        case NARROW:
            if theta_dot > SWITCH_THRESHOLD_HIGH  // 30°/s
               and持续 > 0.5s:
                eo_status.fov_mode = WIDE
                // 在广角图像中计算搜索窗口
                search_roi = predict_target_in_wide_fov(track)
                eo_command.switch_to_wide(search_roi)

            elif track_lost_duration > 0.5s:
                eo_status.fov_mode = WIDE
                eo_command.switch_to_wide(full_frame)

        case WIDE:
            if theta_dot < SWITCH_THRESHOLD_LOW  // 10°/s
               and 持续 > 1.0s
               and track.range > 300m:
                eo_status.fov_mode = NARROW
                eo_command.switch_to_narrow(track.az, track.el)

    return eo_status
```

### 5.4 场景 S4：蛙跳机动

**场景特征**：
- 目标交替执行飞行-悬停-飞行
- 悬停时雷达多普勒检测失效（零速）
- 蛙跳目的是规避雷达探测

**核心挑战**：悬停阶段雷达丢失

**传感器角色**：
- 雷达：飞行阶段有效，悬停阶段微多普勒辅助
- 光电：全程有效（不依赖多普勒）
- 声学：悬停阶段最可靠（旋翼噪声不停）
- RF：全程辅助

**蛙跳检测算法**：

```
function detect_frog_leap(track):
    // 基于IMM模型概率序列检测蛙跳
    history = track.model_prob_history[-20:]  // 最近20帧

    // 计算CV和Hover模型的切换次数
    transitions = 0
    prev_dominant = argmax(history[0])
    for t in range(1, len(history)):
        curr_dominant = argmax(history[t])
        if prev_dominant != curr_dominant:
            if (prev_dominant == CV and curr_dominant == HOVER) or
               (prev_dominant == HOVER and curr_dominant == CV):
                transitions += 1
        prev_dominant = curr_dominant

    // 蛙跳判定: 4秒内 CV↔Hover 切换 >= 2 次
    if transitions >= 2:
        track.frog_leap_flag = True
        return True

    return False
```

**蛙跳融合流程**：

```
┌────────────────────────────────────────────────────────────┐
│ S4: 蛙跳融合流程                                           │
│                                                            │
│  蛙跳检测 ──→ 是蛙跳? ──→ 激活蛙跳模式                    │
│                                                            │
│  蛙跳模式激活后的策略调整:                                 │
│                                                            │
│  (1) IMM转移矩阵切换为 Pi_frog:                            │
│      CV→Hover: 0.05 → 0.15                                │
│      Hover→CV: 0.10 → 0.20                                │
│                                                            │
│  (2) 航迹外推时间延长:                                     │
│      N_coast_start: 20帧 → 40帧 (8秒)                     │
│      保护悬停期间不丢失航迹                                │
│                                                            │
│  (3) 雷达微多普勒检测增强:                                 │
│      悬停检测到IMM Hover概率 > 0.5时,                      │
│      自动在该方向上启动STFT微多普勒分析                    │
│      检测旋翼微多普勒特征                                  │
│                                                            │
│  (4) 声学优先级提升:                                       │
│      W_acou: 0.10 → 0.35                                  │
│      悬停时旋翼噪声稳定, 声学最可靠                        │
│                                                            │
│  (5) 分阶段处理:                                           │
│                                                            │
│  飞行阶段 (mu_CV > 0.6):                                   │
│    ├─ 雷达正常检测 + 光电跟踪                              │
│    ├─ 声学辅助角度                                         │
│    └─ 标准序贯更新                                         │
│                                                            │
│  悬停阶段 (mu_Hover > 0.6):                                │
│    ├─ 雷达: 微多普勒辅助检测(可能仍无法检测)              │
│    ├─ 声学: 主传感器, DOA连续跟踪                          │
│    ├─ 光电: 持续跟踪(不依赖多普勒)                        │
│    └─ 雷达无量测时: 光电+声学纯角度更新                   │
│                                                            │
│  过渡阶段 (模型概率切换中):                                │
│    ├─ 增大过程噪声(允许快速加速/减速)                     │
│    ├─ 放宽关联门限                                        │
│    └─ 多传感器并行搜索                                    │
│                                                            │
│  (6) 蛙跳轨迹预测:                                         │
│      悬停点记录 → 分析蛙跳间距和方向                       │
│      预测下一次飞行方向 → 引导光电预瞄                     │
└────────────────────────────────────────────────────────────┘
```

**蛙跳轨迹预测**：

```
function predict_frog_trajectory(track):
    hover_points = track.hover_history  // 历史悬停位置
    if len(hover_points) >= 2:
        // 计算蛙跳方向向量
        last_hop = hover_points[-1] - hover_points[-2]
        hop_direction = last_hop / norm(last_hop)
        hop_distance = norm(last_hop)

        // 预测下一个悬停点
        next_hover_pred = hover_points[-1] + hop_direction * hop_distance
        uncertainty = hop_distance * 0.3  // 30%不确定性

        // 引导光电预瞄
        predict_az = atan2(next_hover_pred.e, next_hover_pred.n)
        predict_el = atan2(next_hover_pred.u,
                          sqrt(next_hover_pred.e^2 + next_hover_pred.n^2))

        return predict_az, predict_el, uncertainty

    return None
```

### 5.5 场景 S5：城市复杂背景

**场景特征**：
- 楼宇密集，存在大量遮挡
- 雷达多径效应产生虚假目标
- 光电视线被建筑物遮挡
- 目标在楼宇间穿梭

**核心挑战**：间歇性遮挡和多径干扰

**传感器角色**：
- 雷达：受限（多径、遮挡），需结合地图使用
- 光电：受限（遮挡），非遮挡区域内有效
- 声学：优势传感器（声波可绕射建筑物，~300m内有效）
- RF：优势传感器（电磁波可穿透/绕射）

**融合流程**：

```
┌────────────────────────────────────────────────────────────┐
│ S5: 城市融合流程                                           │
│                                                            │
│  环境初始化:                                               │
│  ┌──────────────────────────────────────┐                  │
│  │ 加载城市数字高程模型(DEM) + 建筑模型  │                  │
│  │ 计算各方向的遮挡图:                  │                  │
│  │   LoS_radar(Az, El) → 0/1           │                  │
│  │   LoS_eo(Az, El) → 0/1             │                  │
│  │ 标记多径高风险区域                   │                  │
│  └──────────────────────────────────────┘                  │
│                                                            │
│  实时融合策略:                                             │
│                                                            │
│  (1) 传感器有效性实时判断:                                 │
│      for each track:                                       │
│        track_az, track_el = track.predicted_angle()        │
│        radar_usable = LoS_radar(track_az, track_el)        │
│        eo_usable = LoS_eo(track_az, track_el)              │
│        acoustic_usable = True (声学始终可用)               │
│        rf_usable = True (RF始终可用)                       │
│                                                            │
│  (2) 非遮挡区域 → 标准融合:                               │
│      if radar_usable and eo_usable:                        │
│        标准 S1 流程                                        │
│                                                            │
│  (3) 遮挡区域 → 声学+RF主导:                              │
│      if not radar_usable or not eo_usable:                 │
│        声学DOA + RF测向 → 纯角度跟踪                      │
│        IMM预测维持航迹位置估计                             │
│        航迹外推时间延长(允许穿越遮挡区)                    │
│                                                            │
│  (4) 目标出遮挡 → 快速重关联:                             │
│      目标从遮挡区出来后:                                   │
│        雷达/光电重新检测到目标                             │
│        放宽关联门限(因为外推位置不精确)                    │
│        重新关联到外推航迹                                  │
│        协方差重置(外推段精度差)                            │
│                                                            │
│  (5) 多径虚假目标抑制:                                     │
│      对雷达点迹:                                           │
│        if 点迹位置在已知多径区域 and                       │
│           无光电/声学确认:                                  │
│          标记为疑似多径, 不建立航迹                        │
│        else:                                               │
│          正常处理                                           │
│                                                            │
│  (6) 地图辅助航迹预测:                                     │
│      在航迹预测中加入建筑物约束:                           │
│        if 预测位置在建筑物内:                              │
│          修正预测到建筑物边缘                               │
│          增大位置不确定性                                   │
│          考虑目标绕建筑物运动的可能路径                     │
└────────────────────────────────────────────────────────────┘
```

**地图辅助航迹约束**：

```
function map_constrained_predict(track, building_model):
    // 标准IMM预测
    x_pred = F * track.x_fused

    // 检查预测位置是否在建筑物内
    if building_model.is_inside(x_pred.pos):
        // 找到最近的建筑物边界点
        boundary_point = building_model.nearest_boundary(x_pred.pos)

        // 修正预测位置
        x_pred.pos = boundary_point

        // 增大协方差(反映约束引入的额外不确定性)
        P_pred *= 2.0

        // 生成候选运动路径(绕建筑物)
        candidate_paths = building_model.compute_detour_paths(
            track.x_fused.pos, x_pred.pos, max_paths=3)

        // 将候选路径作为额外的IMM模型处理
        // (简化实现: 仅对预测协方差进行方向性展宽)

    return x_pred, P_pred
```

### 5.6 场景 S6：恶劣天气

**场景特征**：
- 降雨/浓雾导致能见度 < 1 km
- 光电系统严重受限
- 雷达受雨衰影响但基本可用
- 声学受降雨噪声影响

**融合策略**：

```
S6 权重调整:
  W_radar: 0.35 (基本可用,主传感器)
  W_eo:    0.10 (严重受限)
  W_acou:  0.30 (降雨噪声有影响但仍可用)
  W_rf:    0.25 (不受天气影响)

特殊处理:
  · 光电切换到红外通道(对雨雾有一定穿透能力)
  · 声学降噪增强(自适应噪声估计+对消)
  · 雷达考虑雨衰补偿
  · 降低整体虚警率要求(避免因天气导致大量虚警)
```

### 5.7 场景 S7：自主飞行（无 RF 信号）

**场景特征**：
- 目标为自主导航无人机，无遥控信号
- RF 侦测完全失效

**融合策略**：

```
S7 权重调整:
  W_radar: 0.35
  W_eo:    0.35
  W_acou:  0.30
  W_rf:    0.00 (完全排除)

特殊处理:
  · RF传感器自动降权至0
  · 声学重要性提升(唯一的被动近距探测手段)
  · 航迹确认仍需2个传感器(雷达+光电 或 雷达+声学)
  · 分类融合中RF的BPA设为全不确定 m(Θ)=1
```

### 5.8 场景 S8：多目标密集

**场景特征**：
- 同时跟踪 > 5 批目标
- 量测-航迹关联复杂度急剧上升
- 可能出现目标交叉（航迹混淆）

**融合策略**：

```
S8 策略调整:
  · JPDA降级为GNN(全局最近邻): 降低计算复杂度
  · 收紧关联门限: 减少错误关联
  · 光电采用多目标检测模式(YOLO全图检测)
  · 声学采用宽带MUSIC(同时估计多个DOA)
  · 航迹优先级管理:
    - 威胁等级高的目标: 高更新率
    - 威胁等级低的目标: 标准更新率
  · 资源分配:
    - 光电云台优先跟踪最高威胁目标
    - 雷达优先扫描高威胁方向
```

---

## 6 传感器协同管理

### 6.1 光电引导策略

光电引导是最重要的传感器协同策略，决定光电云台的指向：

```
function eo_guidance_manager(tracks, eo_status):
    // 优先级排序
    priority_list = []
    for track in tracks:
        priority = compute_eo_priority(track)
        priority_list.append((track, priority))

    sort(priority_list, by=priority, descending=True)

    // 工作模式选择
    if eo_status.mode == SEARCH:
        // 搜索模式: 引导至最高优先级的未确认暂态航迹
        for track, _ in priority_list:
            if track.status == TENTATIVE:
                guide_eo_to(track.predicted_az, track.predicted_el)
                break
        else:
            // 无暂态航迹: 扫描巡视模式
            eo_scan_pattern()

    elif eo_status.mode == TRACK:
        // 跟踪模式: 持续跟踪最高优先级已确认航迹
        target_track = priority_list[0][0]
        if target_track.status == CONFIRMED:
            // 光电自主跟踪
            pass
        else:
            // 丢失后重新引导
            guide_eo_to(target_track.predicted_az,
                        target_track.predicted_el)

function compute_eo_priority(track):
    // 优先级计算
    priority = 0
    priority += track.threat_score * 30          // 威胁分值
    priority += (1 - track.range / 3000) * 20    // 距离近优先
    priority += (2 - track.confirm_sensor_cnt) * 15  // 未确认优先
    priority += (1 if track.status == TENTATIVE else 0) * 10
    return priority
```

### 6.2 雷达工作模式管理

```
function radar_mode_manager(scene, tracks):
    if S2 in scene:  // 超低空
        radar.enable_clutter_map_enhanced()
        radar.set_micro_doppler_search(True)

    if S4 in scene:  // 蛙跳
        // 在悬停目标方向执行微多普勒检测
        hover_tracks = [t for t in tracks if t.prob_hover > 0.5]
        for t in hover_tracks:
            radar.add_micro_doppler_beam(t.azimuth, t.elevation)

    if S8 in scene:  // 多目标
        radar.set_scan_rate(HIGH)  // 提高扫描速率

    if S3 in scene:  // 近距
        radar.set_mode(NEAR_RANGE_HIGH_RES)
```

### 6.3 融合更新率管理

```
function update_rate_manager(tracks):
    for track in tracks:
        if track.threat_level >= RED:
            track.update_rate = 50  // Hz, 打击引导
        elif track.threat_level >= ORANGE:
            track.update_rate = 20  // Hz, 准备打击
        elif track.status == TENTATIVE:
            track.update_rate = 10  // Hz, 快速确认
        else:
            track.update_rate = 5   // Hz, 标准监视
```

---

## 7 场景切换与平滑过渡

### 7.1 场景切换条件

场景切换采用 **滞回（hysteresis）** 机制，避免频繁切换：

```
function scene_transition(current_scene, new_scene_raw,
                          transition_count):
    // 需要连续 N 帧检测到新场景才切换
    N_switch = 5  // 5帧 = 1秒

    if new_scene_raw != current_scene:
        transition_count += 1
        if transition_count >= N_switch:
            // 确认切换
            current_scene = new_scene_raw
            transition_count = 0
    else:
        transition_count = 0

    return current_scene, transition_count
```

### 7.2 权重平滑过渡

场景切换时，传感器权重采用指数平滑过渡：

\[
w_s(k) = \tau \cdot w_s^{new} + (1 - \tau) \cdot w_s(k-1)
\]

\(\tau = 0.3\)（约 3 帧完成 ~95% 过渡），避免权重突变导致航迹跳变。

---

## 8 场景融合策略总结

```
┌─────────────────────────────────────────────────────────┐
│               场景自适应融合策略总览                      │
│                                                         │
│  ┌───────────────┐                                      │
│  │ 环境感知输入   │                                      │
│  │ · 雷达杂波     │                                      │
│  │ · 能见度       │                                      │
│  │ · 声学SNR     │                                      │
│  │ · RF信号      │                                      │
│  │ · 地形类型     │                                      │
│  │ · IMM模型概率  │                                      │
│  │ · 目标距离/高度│                                      │
│  └───────┬───────┘                                      │
│          │                                              │
│  ┌───────▼───────┐                                      │
│  │ 场景分类器     │                                      │
│  │ (规则引擎)     │                                      │
│  └───────┬───────┘                                      │
│          │                                              │
│  ┌───────▼───────┐      ┌────────────────┐              │
│  │ 先验权重查表   │─────▶│ 权重融合       │              │
│  └───────────────┘      │                │              │
│                         │ 先验 × 残差自   │              │
│  ┌───────────────┐      │ 适应调整       │              │
│  │ 残差反馈       │─────▶│                │              │
│  └───────────────┘      └───────┬────────┘              │
│                                 │                       │
│                         ┌───────▼────────┐              │
│                         │ 自适应R矩阵    │              │
│                         │ R_s / w_s      │              │
│                         └───────┬────────┘              │
│                                 │                       │
│                         ┌───────▼────────┐              │
│                         │ IMM-UKF        │              │
│                         │ 序贯更新       │              │
│                         └───────┬────────┘              │
│                                 │                       │
│                         ┌───────▼────────┐              │
│                         │ 融合航迹输出    │              │
│                         └────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

---

*文档结束*
