# 核心融合算法详细设计

## 1 文档信息

| 项目 | 内容 |
|------|------|
| 文档编号 | CUAV-ALG-004 |
| 版本 | V1.0.0 |
| 日期 | 2026-02-20 |

---

## 2 概述

本文档详细描述多模态融合系统中的核心算法设计，包括：
1. 联合概率数据关联（JPDA）算法
2. 交互多模型-无迹卡尔曼滤波（IMM-UKF）算法
3. 异构传感器序贯更新策略
4. D-S 证据理论分类融合算法
5. 航迹管理算法

所有算法均给出完整的数学推导和伪代码实现。

---

## 3 状态空间模型

### 3.1 状态向量定义

目标状态向量为 6 维，包含三维位置和三维速度（ENU 坐标系）：

\[
\mathbf{x} = [x_E, \dot{x}_E, x_N, \dot{x}_N, x_U, \dot{x}_U]^T
\]

### 3.2 运动模型集

系统采用三种运动模型，覆盖低慢小目标的典型运动模式：

#### 模型 M1：匀速直线 (CV)

\[
\mathbf{x}_{k+1} = \mathbf{F}_{CV} \mathbf{x}_k + \mathbf{G} \mathbf{w}_k
\]

\[
\mathbf{F}_{CV} = \begin{bmatrix} 1 & T & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & T & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & T \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}
\]

\[
\mathbf{G} = \begin{bmatrix} T^2/2 & 0 & 0 \\ T & 0 & 0 \\ 0 & T^2/2 & 0 \\ 0 & T & 0 \\ 0 & 0 & T^2/2 \\ 0 & 0 & T \end{bmatrix}
\]

过程噪声协方差：

\[
\mathbf{Q}_{CV} = \mathbf{G} \cdot \text{diag}(\sigma_{a,CV}^2, \sigma_{a,CV}^2, \sigma_{a,CV}^2) \cdot \mathbf{G}^T
\]

其中 \(\sigma_{a,CV} = 1\) m/s²（低慢小目标机动性较低）

#### 模型 M2：匀速转弯 (CT)

\[
\mathbf{F}_{CT}(\omega) = \begin{bmatrix} 1 & \frac{\sin\omega T}{\omega} & 0 & -\frac{1-\cos\omega T}{\omega} & 0 & 0 \\ 0 & \cos\omega T & 0 & -\sin\omega T & 0 & 0 \\ 0 & \frac{1-\cos\omega T}{\omega} & 1 & \frac{\sin\omega T}{\omega} & 0 & 0 \\ 0 & \sin\omega T & 0 & \cos\omega T & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & T \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}
\]

转弯率 \(\omega\) 根据上一帧航迹估计：

\[
\omega = \frac{\dot{x}_E \ddot{x}_N - \dot{x}_N \ddot{x}_E}{\dot{x}_E^2 + \dot{x}_N^2}
\]

若无法估计，取固定值 \(\omega_0 = 5°/s = 0.087\) rad/s。

过程噪声：

\[
\mathbf{Q}_{CT} = \mathbf{G} \cdot \text{diag}(\sigma_{a,CT}^2, \sigma_{a,CT}^2, \sigma_{a,CT}^2) \cdot \mathbf{G}^T
\]

其中 \(\sigma_{a,CT} = 3\) m/s²（转弯时机动性较高）

#### 模型 M3：悬停/静止 (Hover)

\[
\mathbf{F}_{hover} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & \alpha_v & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & \alpha_v & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & \alpha_v \end{bmatrix}
\]

其中 \(\alpha_v = 0.05\)（速度快速衰减至零，但不硬置零以避免
数值奇异）。

过程噪声：

\[
\mathbf{Q}_{hover} = \text{diag}(0.1^2, 0.5^2, 0.1^2, 0.5^2, 0.1^2, 0.3^2)
\]

位置噪声小（悬停基本不动），速度噪声适中（允许小幅漂移）。

### 3.3 量测模型

#### 雷达量测模型（4 维）

\[
\mathbf{z}_{radar} = h_{radar}(\mathbf{x}) + \mathbf{v}_{radar}
\]

\[
h_{radar}(\mathbf{x}) = \begin{bmatrix} \sqrt{x_E^2 + x_N^2 + x_U^2} \\ \arctan(x_E / x_N) \\ \arctan(x_U / \sqrt{x_E^2 + x_N^2}) \\ (x_E \dot{x}_E + x_N \dot{x}_N + x_U \dot{x}_U) / R \end{bmatrix} = \begin{bmatrix} R \\ Az \\ El \\ \dot{R} \end{bmatrix}
\]

量测噪声：

\[
\mathbf{R}_{radar} = \text{diag}(\sigma_R^2, \sigma_{Az,r}^2, \sigma_{El,r}^2, \sigma_{\dot{R}}^2) = \text{diag}(4, 2.5 \times 10^{-3}, 2.5 \times 10^{-3}, 0.09)
\]

（单位：m², rad², rad², (m/s)²）

#### 光电量测模型（2 维）

\[
h_{EO}(\mathbf{x}) = \begin{bmatrix} \arctan(x_E / x_N) \\ \arctan(x_U / \sqrt{x_E^2 + x_N^2}) \end{bmatrix} = \begin{bmatrix} Az \\ El \end{bmatrix}
\]

量测噪声：

\[
\mathbf{R}_{EO} = \text{diag}(\sigma_{Az,eo}^2, \sigma_{El,eo}^2) = \text{diag}(2.5 \times 10^{-7}, 2.5 \times 10^{-7})
\]

（0.5 mrad 精度）

注意：广角相机精度降低，使用：

\[
\mathbf{R}_{EO,wide} = \text{diag}(7.6 \times 10^{-5}, 7.6 \times 10^{-5})
\]

（~0.5° 精度）

#### 声学量测模型（2 维）

\[
h_{acou}(\mathbf{x}) = \begin{bmatrix} \arctan(x_E / x_N) \\ \arctan(x_U / \sqrt{x_E^2 + x_N^2}) \end{bmatrix}
\]

量测噪声：

\[
\mathbf{R}_{acou} = \text{diag}(\sigma_{Az,ac}^2, \sigma_{El,ac}^2) = \text{diag}(2.7 \times 10^{-3}, 1.95 \times 10^{-2})
\]

（3° 方位, 8° 俯仰）

#### RF 量测模型（1 维）

\[
h_{RF}(\mathbf{x}) = \arctan(x_E / x_N) = Az
\]

量测噪声：

\[
R_{RF} = \sigma_{Az,rf}^2 = 7.6 \times 10^{-3}
\]

（5° 精度）

---

## 4 联合概率数据关联 (JPDA)

### 4.1 算法概述

JPDA 是一种软关联算法，允许一个量测以一定概率同时关联到多条航迹，
避免了硬决策（最近邻）的错误传播问题。

### 4.2 算法流程

**输入**：
- 当前周期量测集 \(\mathbf{Z} = \{\mathbf{z}_1, \ldots, \mathbf{z}_m\}\)
- 航迹集 \(\mathcal{T} = \{T_1, \ldots, T_n\}\)，每条航迹有预测状态
  \(\hat{\mathbf{x}}_j^-\) 和预测协方差 \(\mathbf{P}_j^-\)

**输出**：
- 关联概率矩阵 \(\{\beta_{ij}\}\)
- 未关联量测列表

#### 步骤 1：预测量测和新息协方差

对每条航迹 \(T_j\)，计算预测量测和新息协方差。由于使用 UKF，
通过 Sigma 点传播：

**生成 Sigma 点**：

\[
\chi_0 = \hat{\mathbf{x}}_j^-
\]

\[
\chi_i = \hat{\mathbf{x}}_j^- + \sqrt{(n_x + \lambda) \mathbf{P}_j^-} \big|_i, \quad i = 1, \ldots, n_x
\]

\[
\chi_{i+n_x} = \hat{\mathbf{x}}_j^- - \sqrt{(n_x + \lambda) \mathbf{P}_j^-} \big|_i, \quad i = 1, \ldots, n_x
\]

其中 \(n_x = 6\), \(\lambda = \alpha^2(n_x + \kappa) - n_x\),
取 \(\alpha = 0.01\), \(\beta_{UKF} = 2\), \(\kappa = 0\)。

**量测预测**：

\[
\hat{\mathbf{z}}_j = \sum_{i=0}^{2n_x} W_i^{(m)} h_s(\chi_i)
\]

其中 \(h_s\) 为对应传感器的量测函数。

**新息协方差**：

\[
\mathbf{S}_j = \sum_{i=0}^{2n_x} W_i^{(c)} (\hat{\mathbf{z}}_i - \hat{\mathbf{z}}_j)(\hat{\mathbf{z}}_i - \hat{\mathbf{z}}_j)^T + \mathbf{R}_s
\]

#### 步骤 2：关联门限检验

对每个量测-航迹对 \((i, j)\)，计算马氏距离：

\[
d_{ij}^2 = (\mathbf{z}_i - \hat{\mathbf{z}}_j)^T \mathbf{S}_j^{-1} (\mathbf{z}_i - \hat{\mathbf{z}}_j)
\]

如果 \(d_{ij}^2 \leq \gamma\)，则该对通过门限。

**门限值选择**（基于卡方分布，\(P_G = 0.997\)）：

| 量测维度 | 门限 γ |
|---------|--------|
| 1 (RF) | 8.81 |
| 2 (光电/声学) | 11.83 |
| 4 (雷达) | 16.27 |

#### 步骤 3：构造验证矩阵

构造二进制验证矩阵 \(\Omega\)，其中 \(\Omega_{ij} = 1\) 表示量测
\(i\) 落入航迹 \(j\) 的关联门限内。

```
        T1  T2  T3  ...  Tn
z1    [ 1   0   0  ...   0 ]
z2    [ 1   1   0  ...   0 ]
z3    [ 0   0   1  ...   0 ]
...
zm    [ 0   0   0  ...   1 ]
```

#### 步骤 4：枚举可行联合事件

一个联合事件 \(\theta\) 是量测到航迹的一个分配方案，满足：
- 每个量测最多关联到一条航迹
- 每条航迹最多关联到一个量测

可行联合事件集 \(\Theta\) 是所有满足上述约束且与验证矩阵一致的
分配方案的集合。

**实现优化**：当目标数和量测数较大时，使用分簇（clustering）
策略将独立子问题分开处理，降低枚举复杂度。

#### 步骤 5：计算联合事件概率

每个联合事件 \(\theta\) 的概率：

\[
P(\theta | \mathbf{Z}) \propto \prod_{j: \theta(j) \neq 0} \frac{p_{ij}}{P_D \cdot \rho} \prod_{j: \theta(j) = 0} (1 - P_D)
\]

其中：
- \(p_{ij} = \mathcal{N}(\mathbf{z}_i; \hat{\mathbf{z}}_j, \mathbf{S}_j)\) 为似然
- \(P_D\) 为检测概率（雷达 0.8，光电 0.9，声学 0.7，RF 0.6）
- \(\rho\) 为杂波密度（量测空间中的均匀杂波率）

#### 步骤 6：计算边缘关联概率

量测 \(i\) 关联到航迹 \(j\) 的边缘概率：

\[
\beta_{ij} = \sum_{\theta \in \Theta: \theta(j)=i} P(\theta | \mathbf{Z})
\]

航迹 \(j\) 无关联量测的概率：

\[
\beta_{0j} = \sum_{\theta \in \Theta: \theta(j)=0} P(\theta | \mathbf{Z})
\]

满足 \(\beta_{0j} + \sum_{i=1}^{m} \beta_{ij} = 1\)

#### 步骤 7：加权新息计算

合并新息：

\[
\mathbf{v}_j = \sum_{i=1}^{m} \beta_{ij} (\mathbf{z}_i - \hat{\mathbf{z}}_j) = \sum_{i=1}^{m} \beta_{ij} \mathbf{v}_{ij}
\]

加权更新（传递给 UKF 更新步骤）。

### 4.3 伪代码

```
function JPDA(tracks, measurements, params):
    // Step 1: 预测量测
    for each track T_j in tracks:
        z_pred_j, S_j = ukf_predict_measurement(T_j, sensor_type)

    // Step 2: 关联门限检验
    validation_matrix = zeros(m, n)
    for i = 1 to m:
        for j = 1 to n:
            d2 = mahalanobis_distance(z[i], z_pred[j], S[j])
            if d2 <= gamma:
                validation_matrix[i][j] = 1

    // Step 3: 分簇
    clusters = find_independent_clusters(validation_matrix)

    // Step 4-6: 对每个簇独立处理
    beta = zeros(m+1, n)  // beta[0][j] = 无关联概率
    for each cluster C in clusters:
        events = enumerate_joint_events(C, validation_matrix)
        for each event theta in events:
            prob = compute_event_probability(theta, ...)
        for j in C.tracks:
            for i in C.measurements:
                beta[i][j] = marginal_probability(i, j, events)
            beta[0][j] = 1.0 - sum(beta[1:m][j])

    // Step 7: 计算加权新息
    for each track T_j:
        v_combined_j = sum(beta[i][j] * (z[i] - z_pred[j])
                          for i with beta[i][j] > 0)

    // 未关联量测
    unassociated = [z_i for i if max(beta[i][:]) < threshold]

    return beta, v_combined, unassociated
```

---

## 5 交互多模型-无迹卡尔曼滤波 (IMM-UKF)

### 5.1 算法概述

IMM 算法维护多个并行的运动模型滤波器，通过模型概率加权输出最终
估计。每个模型内部使用 UKF 处理非线性量测方程。

### 5.2 马尔可夫模型转移矩阵

\[
\Pi = [\pi_{ij}]_{3 \times 3} = \begin{bmatrix} 0.90 & 0.05 & 0.05 \\ 0.05 & 0.90 & 0.05 \\ 0.10 & 0.05 & 0.85 \end{bmatrix}
\]

含义：
- \(\pi_{ij}\) = 从模型 i 转移到模型 j 的概率
- CV 模型有 90% 概率维持，5% 转 CT，5% 转 Hover
- Hover 模型有 10% 概率转 CV（蛙跳起飞），较高

**蛙跳场景增强**：检测到蛙跳行为后，动态调整：

\[
\Pi_{frog} = \begin{bmatrix} 0.80 & 0.05 & 0.15 \\ 0.05 & 0.85 & 0.10 \\ 0.20 & 0.05 & 0.75 \end{bmatrix}
\]

增大 CV ↔ Hover 之间的转移概率。

### 5.3 IMM 完整算法流程

每个融合周期执行以下四个步骤：

#### 步骤 1：交互/混合 (Interaction/Mixing)

**目的**：将上一时刻各模型的估计按转移概率混合，作为当前时刻各
模型的初始条件。

计算混合权重：

\[
\bar{c}_j = \sum_{i=1}^{r} \pi_{ij} \mu_i(k-1)
\]

\[
\mu_{i|j} = \frac{\pi_{ij} \mu_i(k-1)}{\bar{c}_j}
\]

混合状态估计：

\[
\hat{\mathbf{x}}_{0j}(k-1) = \sum_{i=1}^{r} \mu_{i|j} \hat{\mathbf{x}}_i(k-1)
\]

混合协方差：

\[
\mathbf{P}_{0j}(k-1) = \sum_{i=1}^{r} \mu_{i|j} \left[\mathbf{P}_i(k-1) + (\hat{\mathbf{x}}_i(k-1) - \hat{\mathbf{x}}_{0j}(k-1))(\hat{\mathbf{x}}_i(k-1) - \hat{\mathbf{x}}_{0j}(k-1))^T\right]
\]

#### 步骤 2：模型条件 UKF 滤波

对每个模型 \(j = 1, 2, 3\) 独立执行 UKF：

##### 2a. UKF 时间预测

**Sigma 点生成**（2L+1 个，L = 6）：

\[
\mathcal{X}_0 = \hat{\mathbf{x}}_{0j}
\]

\[
\mathcal{X}_i = \hat{\mathbf{x}}_{0j} + \left(\sqrt{(L+\lambda)\mathbf{P}_{0j}}\right)_i, \quad i = 1, \ldots, L
\]

\[
\mathcal{X}_{i+L} = \hat{\mathbf{x}}_{0j} - \left(\sqrt{(L+\lambda)\mathbf{P}_{0j}}\right)_i, \quad i = 1, \ldots, L
\]

权重：

\[
W_0^{(m)} = \frac{\lambda}{L+\lambda}, \quad W_0^{(c)} = \frac{\lambda}{L+\lambda} + (1-\alpha^2+\beta_{UKF})
\]

\[
W_i^{(m)} = W_i^{(c)} = \frac{1}{2(L+\lambda)}, \quad i = 1, \ldots, 2L
\]

**状态传播**：

\[
\mathcal{X}_i^* = f_j(\mathcal{X}_i)
\]

其中 \(f_j\) 为模型 j 的状态转移函数。

**预测状态和协方差**：

\[
\hat{\mathbf{x}}_j^- = \sum_{i=0}^{2L} W_i^{(m)} \mathcal{X}_i^*
\]

\[
\mathbf{P}_j^- = \sum_{i=0}^{2L} W_i^{(c)} (\mathcal{X}_i^* - \hat{\mathbf{x}}_j^-)(\mathcal{X}_i^* - \hat{\mathbf{x}}_j^-)^T + \mathbf{Q}_j
\]

##### 2b. UKF 量测更新（序贯，参见第 6 节）

**重新生成 Sigma 点**（基于预测状态）：

\[
\mathcal{X}_i^{pred} \text{ from } (\hat{\mathbf{x}}_j^-, \mathbf{P}_j^-)
\]

**量测预测**：

\[
\hat{\mathbf{z}}_s = \sum_{i=0}^{2L} W_i^{(m)} h_s(\mathcal{X}_i^{pred})
\]

**新息协方差**：

\[
\mathbf{S}_{j,s} = \sum_{i=0}^{2L} W_i^{(c)} (h_s(\mathcal{X}_i^{pred}) - \hat{\mathbf{z}}_s)(h_s(\mathcal{X}_i^{pred}) - \hat{\mathbf{z}}_s)^T + \mathbf{R}_s
\]

**互协方差**：

\[
\mathbf{P}_{xz,j,s} = \sum_{i=0}^{2L} W_i^{(c)} (\mathcal{X}_i^{pred} - \hat{\mathbf{x}}_j^-)(h_s(\mathcal{X}_i^{pred}) - \hat{\mathbf{z}}_s)^T
\]

**卡尔曼增益**：

\[
\mathbf{K}_{j,s} = \mathbf{P}_{xz,j,s} \mathbf{S}_{j,s}^{-1}
\]

**使用 JPDA 加权新息更新**：

\[
\hat{\mathbf{x}}_j = \hat{\mathbf{x}}_j^- + \mathbf{K}_{j,s} \mathbf{v}_j^{combined}
\]

**协方差更新**（考虑关联不确定性）：

\[
\mathbf{P}_j = \beta_{0j} \mathbf{P}_j^- + (1 - \beta_{0j})(\mathbf{P}_j^- - \mathbf{K}_{j,s} \mathbf{S}_{j,s} \mathbf{K}_{j,s}^T) + \tilde{\mathbf{P}}_j
\]

其中扩展项：

\[
\tilde{\mathbf{P}}_j = \mathbf{K}_{j,s} \left[\sum_{i=1}^{m} \beta_{ij} \mathbf{v}_{ij} \mathbf{v}_{ij}^T - \mathbf{v}_j^{combined} (\mathbf{v}_j^{combined})^T\right] \mathbf{K}_{j,s}^T
\]

##### 2c. 模型似然计算

\[
\Lambda_j = \mathcal{N}(\mathbf{v}_j^{combined}; \mathbf{0}, \mathbf{S}_{j,s})
\]

\[
= \frac{1}{(2\pi)^{n_z/2} |\mathbf{S}_{j,s}|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{v}_j^{combined})^T \mathbf{S}_{j,s}^{-1} \mathbf{v}_j^{combined}\right)
\]

#### 步骤 3：模型概率更新

\[
\mu_j(k) = \frac{\Lambda_j \bar{c}_j}{\sum_{i=1}^{r} \Lambda_i \bar{c}_i}
\]

#### 步骤 4：估计融合

加权融合输出：

\[
\hat{\mathbf{x}}(k) = \sum_{j=1}^{r} \mu_j(k) \hat{\mathbf{x}}_j(k)
\]

\[
\mathbf{P}(k) = \sum_{j=1}^{r} \mu_j(k) \left[\mathbf{P}_j(k) + (\hat{\mathbf{x}}_j(k) - \hat{\mathbf{x}}(k))(\hat{\mathbf{x}}_j(k) - \hat{\mathbf{x}}(k))^T\right]
\]

### 5.4 伪代码

```
function IMM_UKF_cycle(track, measurements, params):
    r = 3  // 模型数
    mu = track.model_probs  // [mu_CV, mu_CT, mu_hover]
    Pi = params.transition_matrix

    // =============== Step 1: 交互/混合 ===============
    c_bar = zeros(r)
    for j = 1 to r:
        c_bar[j] = sum(Pi[i][j] * mu[i] for i = 1 to r)

    mu_mix = zeros(r, r)
    for i = 1 to r:
        for j = 1 to r:
            mu_mix[i][j] = Pi[i][j] * mu[i] / c_bar[j]

    x0 = zeros(r, 6)   // 混合初始状态
    P0 = zeros(r, 6, 6) // 混合初始协方差
    for j = 1 to r:
        x0[j] = sum(mu_mix[i][j] * track.x[i] for i = 1 to r)
        for i = 1 to r:
            diff = track.x[i] - x0[j]
            P0[j] += mu_mix[i][j] * (track.P[i] + outer(diff, diff))

    // =============== Step 2: UKF 滤波 ===============
    Lambda = zeros(r)
    for j = 1 to r:
        // 时间预测
        x_pred[j], P_pred[j] = ukf_time_predict(
            x0[j], P0[j], F_model[j], Q_model[j])

        // 序贯量测更新 (详见第6节)
        x_upd[j], P_upd[j], Lambda[j] = ukf_sequential_update(
            x_pred[j], P_pred[j], measurements,
            beta_jpda, sensor_weights)

    // =============== Step 3: 模型概率更新 ===============
    c_total = sum(Lambda[j] * c_bar[j] for j = 1 to r)
    for j = 1 to r:
        mu[j] = Lambda[j] * c_bar[j] / c_total

    // =============== Step 4: 融合输出 ===============
    x_fused = sum(mu[j] * x_upd[j] for j = 1 to r)
    P_fused = zeros(6, 6)
    for j = 1 to r:
        diff = x_upd[j] - x_fused
        P_fused += mu[j] * (P_upd[j] + outer(diff, diff))

    // 更新航迹
    track.x_fused = x_fused
    track.P_fused = P_fused
    track.model_probs = mu
    track.x = x_upd
    track.P = P_upd

    return track
```

---

## 6 异构传感器序贯更新

### 6.1 策略说明

由于四种传感器的量测维度不同（雷达 4 维，光电 2 维，声学 2 维，
RF 1 维），采用**序贯更新**策略：每次使用一种传感器的量测执行一次
UKF 量测更新，得到的后验状态作为下一个传感器更新的先验。

**更新顺序**：雷达 → 光电 → 声学 → RF

选择此顺序的理由：
- 雷达提供距离信息（最关键），先更新可大幅收缩状态空间
- 光电角度精度最高，第二步精化角度
- 声学和 RF 角度精度较低，最后更新微调

### 6.2 序贯更新流程

```
输入: 预测状态 x^-, P^-
      各传感器量测 z_radar, z_eo, z_acou, z_rf (可能部分缺失)
      各传感器 JPDA 加权新息

输出: 更新后状态 x, P, 总似然 Lambda

Lambda_total = 1.0

// (1) 雷达量测更新
if z_radar available:
    x1, P1, Lambda_r = ukf_update(x^-, P^-, z_radar,
                                   h_radar, R_radar, w_radar)
    Lambda_total *= Lambda_r
else:
    x1, P1 = x^-, P^-

// (2) 光电量测更新
if z_eo available:
    x2, P2, Lambda_e = ukf_update(x1, P1, z_eo,
                                   h_eo, R_eo, w_eo)
    Lambda_total *= Lambda_e
else:
    x2, P2 = x1, P1

// (3) 声学量测更新
if z_acou available:
    x3, P3, Lambda_a = ukf_update(x2, P2, z_acou,
                                   h_acou, R_acou, w_acou)
    Lambda_total *= Lambda_a
else:
    x3, P3 = x2, P2

// (4) RF量测更新
if z_rf available:
    x4, P4, Lambda_f = ukf_update(x3, P3, z_rf,
                                   h_rf, R_rf, w_rf)
    Lambda_total *= Lambda_f
else:
    x4, P4 = x3, P3

return x4, P4, Lambda_total
```

### 6.3 自适应量测噪声调整

基于场景感知模块输出的传感器可信度权重 \(w_s\)，动态调整量测噪声：

\[
\mathbf{R}_s^{adaptive} = \frac{\mathbf{R}_s}{w_s}
\]

当传感器可信度低时（如雷达在超低空场景），量测噪声增大，相当于
降低该传感器的融合贡献；反之亦然。

### 6.4 纯角度跟踪（无雷达量测场景）

当雷达无法提供量测时（超低空杂波、城市遮挡），系统退化为纯角度
跟踪模式。此时距离不可观测，需要额外约束：

**高度约束法**：

假设目标高度 \(h\) 已知（先验估计或声学/光电辅助判断），
则可通过俯仰角估计距离：

\[
R \approx \frac{h - h_{platform}}{\sin(El)}
\]

**实现方式**：在状态向量中增加高度伪量测：

\[
z_{height} = h_{assumed}, \quad R_{height} = \sigma_h^2
\]

其中 \(\sigma_h\) 取较大值（如 20 m），以反映高度假设的不确定性。

---

## 7 D-S 证据理论目标分类融合

### 7.1 理论框架

**识别框架**：

\[
\Theta = \{\theta_1: \text{多旋翼}, \theta_2: \text{固定翼}, \theta_3: \text{鸟类}, \theta_4: \text{其他}\}
\]

**幂集**：

\[
2^\Theta = \{\emptyset, \{\theta_1\}, \{\theta_2\}, \{\theta_3\}, \{\theta_4\}, \{\theta_1,\theta_2\}, \ldots, \Theta\}
\]

**基本概率分配 (BPA)**：

每个传感器输出一个 BPA 函数 \(m: 2^\Theta \rightarrow [0, 1]\)，
满足 \(m(\emptyset) = 0\) 且 \(\sum_{A \subseteq \Theta} m(A) = 1\)。

### 7.2 各传感器 BPA 生成

#### 雷达 BPA

基于 RCS 大小和微多普勒特征：

```
function radar_bpa(rcs_dBsm, has_micro_doppler):
    if rcs_dBsm > -10 and has_micro_doppler:
        m({多旋翼}) = 0.6
        m({固定翼}) = 0.1
        m(Θ) = 0.3           // 不确定度
    elif rcs_dBsm > -10 and not has_micro_doppler:
        m({固定翼}) = 0.4
        m({多旋翼}) = 0.2
        m(Θ) = 0.4
    elif rcs_dBsm < -20:
        m({鸟类}) = 0.5
        m({其他}) = 0.1
        m(Θ) = 0.4
    else:
        m(Θ) = 1.0           // 完全不确定
    return m
```

#### 光电 BPA

基于 YOLOv8 分类置信度：

```
function eo_bpa(class_id, confidence):
    if confidence > 0.8:
        m({class_id}) = confidence * 0.85
        m(Θ) = 1 - m({class_id})
    elif confidence > 0.5:
        m({class_id}) = confidence * 0.6
        m(Θ) = 1 - m({class_id})
    else:
        m(Θ) = 1.0
    return m
```

#### 声学 BPA

基于 MFCC 分类结果：

```
function acoustic_bpa(class_id, confidence, snr_dB):
    reliability = min(1.0, snr_dB / 20.0)  // SNR越高越可靠
    if confidence > 0.7:
        m({class_id}) = confidence * reliability * 0.7
        m(Θ) = 1 - m({class_id})
    else:
        m(Θ) = 1.0
    return m
```

#### RF BPA

基于协议识别：

```
function rf_bpa(protocol_id, confidence):
    if protocol_id in [DJI, ELRS, DSMX]:
        // 已识别无人机协议
        if protocol_id == DJI:
            m({多旋翼}) = 0.7 * confidence
        else:
            m({多旋翼, 固定翼}) = 0.6 * confidence
        m(Θ) = 1 - sum(m)
    elif protocol_id == WIFI:
        m({多旋翼, 固定翼}) = 0.3 * confidence
        m(Θ) = 1 - m({多旋翼, 固定翼})
    else:
        m(Θ) = 1.0  // 无信号或未识别
    return m
```

### 7.3 D-S 合成规则

两个 BPA \(m_1\) 和 \(m_2\) 的合成：

\[
m_{1,2}(A) = \frac{\sum_{B \cap C = A} m_1(B) \cdot m_2(C)}{1 - K}
\]

其中冲突系数：

\[
K = \sum_{B \cap C = \emptyset} m_1(B) \cdot m_2(C)
\]

**高冲突处理**：当 \(K > 0.8\) 时，表明两个证据源高度矛盾，
此时不进行合成，而是保留各自独立的 BPA，并报告冲突告警。

### 7.4 多传感器序贯合成

```
m_fused = m_radar
m_fused = DS_combine(m_fused, m_eo)
m_fused = DS_combine(m_fused, m_acoustic)
m_fused = DS_combine(m_fused, m_rf)
```

### 7.5 决策规则

基于融合后的 BPA 做出分类决策：

**信任函数和似然函数**：

\[
Bel(A) = \sum_{B \subseteq A} m(B)
\]

\[
Pl(A) = \sum_{B \cap A \neq \emptyset} m(B)
\]

**决策条件**：

1. 选择 \(Bel(\{\theta_i\})\) 最大的类别
2. 且 \(Bel(\{\theta_i\}) > 0.4\)（最低置信门限）
3. 且 \(Bel(\{\theta_i\}) - Bel(\{\theta_j\}) > 0.1\)（与次优拉开差距）
4. 否则判定为"未确认"

### 7.6 伪代码

```
function DS_combine(m1, m2):
    // 计算冲突系数
    K = 0
    for B in power_set(Theta):
        for C in power_set(Theta):
            if B intersect C == empty:
                K += m1(B) * m2(C)

    if K > 0.8:
        return m1  // 冲突过大，不合成

    // 合成
    m_fused = {}
    for A in power_set(Theta):
        if A == empty: continue
        m_fused(A) = 0
        for B in power_set(Theta):
            for C in power_set(Theta):
                if B intersect C == A:
                    m_fused(A) += m1(B) * m2(C)
        m_fused(A) /= (1 - K)

    return m_fused

function classify_target(track, sensor_reports):
    m_radar = radar_bpa(track.rcs, track.micro_doppler_flag)
    m_eo    = eo_bpa(report_eo.class, report_eo.confidence)
    m_acou  = acoustic_bpa(report_ac.class, report_ac.conf,
                           report_ac.snr)
    m_rf    = rf_bpa(report_rf.protocol, report_rf.confidence)

    m = m_radar
    m = DS_combine(m, m_eo)
    m = DS_combine(m, m_acou)
    m = DS_combine(m, m_rf)

    // 决策
    best_class = argmax(Bel(theta_i) for theta_i in Theta)
    if Bel(best_class) > 0.4:
        track.target_class = best_class
        track.class_confidence = Bel(best_class)
    else:
        track.target_class = UNKNOWN
        track.class_confidence = max(Bel(theta_i))

    return track
```

---

## 8 航迹管理算法

### 8.1 航迹起始

#### 8.1.1 单传感器 M/N 逻辑

对每种传感器的未关联量测，独立维护起始候选：

| 传感器 | M/N 参数 | 含义 |
|--------|---------|------|
| 雷达 | 2/3 | 3帧中检测到2次 |
| 光电 | 2/3 | 3帧中检测到2次 |
| 声学 | 4/5 | 5帧中检测到4次 |
| RF | 4/5 | 5帧中检测到4次 |

**M/N 判决实现**：使用滑动窗口位图：

```
function MN_check(det_history, M, N):
    // det_history: 最近 N 帧的检测标志 (ring buffer)
    count = sum(det_history[-N:])
    return count >= M
```

#### 8.1.2 多传感器确认

暂态航迹升级为确认航迹的条件：

```
function confirm_check(tentative_track):
    sensor_types = unique(tentative_track.contributing_sensors)
    if len(sensor_types) >= 2:
        return CONFIRMED
    elif tentative_track.age > 10 frames:
        return TERMINATED  // 超时未确认
    else:
        return TENTATIVE   // 继续等待
```

#### 8.1.3 新航迹状态初始化

根据首次量测类型初始化航迹状态：

**雷达量测初始化**（完整状态）：

\[
\hat{\mathbf{x}}_0 = [R\cos El \sin Az, 0, R\cos El \cos Az, 0, R\sin El, 0]^T
\]

\[
\mathbf{P}_0 = \text{diag}(\sigma_R^2, V_{max}^2/3, \sigma_R^2, V_{max}^2/3, \sigma_R^2, V_{max}^2/3)
\]

初始速度设为零，速度协方差取 \(V_{max}^2/3\) 以覆盖可能的速度范围。

**纯角度量测初始化**（光电/声学/RF）：

需要假设初始距离：

\[
R_{init} = R_{default} = 500 \text{ m (先验估计)}
\]

\[
\sigma_{R,init} = 300 \text{ m (高不确定性)}
\]

### 8.2 航迹维持

**更新计数器**：

```
function update_track_status(track, was_updated):
    if was_updated:
        track.miss_count = 0
        track.update_count += 1
    else:
        track.miss_count += 1

    // 状态转移
    if track.status == CONFIRMED:
        if track.miss_count >= N_coast_start:
            track.status = COASTING

    elif track.status == COASTING:
        if was_updated:
            track.status = CONFIRMED
            track.miss_count = 0
        elif track.miss_count >= N_coast_start + N_coast_end:
            track.status = TERMINATED
```

**参数设置**：

| 参数 | 标准值 | 蛙跳模式值 |
|------|--------|-----------|
| N_coast_start | 20 帧 (4 s) | 40 帧 (8 s) |
| N_coast_end | 5 帧 (1 s) | 10 帧 (2 s) |

### 8.3 航迹终止

```
function check_termination(track):
    // 条件 1: 超时
    if track.status == TERMINATED:
        return DELETE

    // 条件 2: 超出探测范围
    if track.range > R_max * 1.2:
        track.status = TERMINATED
        return DELETE

    // 条件 3: 预测协方差过大 (航迹发散)
    if trace(track.P) > P_max_threshold:
        track.status = TERMINATED
        return DELETE

    // 条件 4: 高度不合理 (地面以下)
    if track.pos_u < -10:
        track.status = TERMINATED
        return DELETE

    return KEEP
```

### 8.4 航迹外推

当航迹进入 COASTING 状态时，仅执行时间预测，不进行量测更新：

```
function coast_track(track):
    // 使用上一次最可能的运动模型进行纯预测
    best_model = argmax(track.model_probs)
    track.x_fused = F[best_model] * track.x_fused
    track.P_fused = F[best_model] * track.P_fused * F[best_model]^T
                    + Q[best_model] * coast_inflate_factor

    // 增大过程噪声以反映外推不确定性
    coast_inflate_factor = 1.0 + 0.5 * track.miss_count

    return track
```

---

## 9 数值稳定性措施

### 9.1 协方差矩阵正定性保证

UKF 更新后可能出现协方差矩阵非正定的情况，采用以下措施：

**Joseph 形式更新**：

\[
\mathbf{P}^+ = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}^-(\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K}\mathbf{R}\mathbf{K}^T
\]

**对称化强制**：

\[
\mathbf{P} \leftarrow \frac{\mathbf{P} + \mathbf{P}^T}{2}
\]

**正定性检查与修复**：

```
function ensure_positive_definite(P):
    P = (P + P^T) / 2  // 对称化
    eigenvalues, eigenvectors = eig(P)
    for i in range(len(eigenvalues)):
        if eigenvalues[i] < epsilon:
            eigenvalues[i] = epsilon  // epsilon = 1e-10
    P = eigenvectors * diag(eigenvalues) * eigenvectors^T
    return P
```

### 9.2 角度回绕处理

方位角量测在 0°/360° 处存在回绕问题：

```
function angle_diff(a, b):
    d = a - b
    while d > pi:
        d -= 2 * pi
    while d < -pi:
        d += 2 * pi
    return d
```

在计算新息时，对角度分量使用此函数。

### 9.3 UKF Sigma 点退化检测

当 Sigma 点传播后过度集中或发散时，进行检测和重置：

```
function check_sigma_spread(sigma_points, x_mean):
    max_dev = max(norm(sigma_points[i] - x_mean)
                  for i in range(len(sigma_points)))
    if max_dev < 1e-8 or max_dev > 1e8:
        // 退化检测: 重置协方差
        return True
    return False
```

---

## 10 算法计算复杂度分析

| 算法模块 | 计算复杂度 | 典型耗时 (10目标, Orin AGX) |
|---------|-----------|---------------------------|
| UKF 时间预测 | O(n³) per model | < 0.1 ms |
| UKF 量测更新 | O(n²m) per sensor | < 0.1 ms |
| IMM 交互 | O(r²n²) | < 0.05 ms |
| IMM 合成 | O(rn²) | < 0.05 ms |
| JPDA 关联 | O(m^n) worst case | < 5 ms (分簇后) |
| D-S 合成 | O(2^|Θ|) per sensor | < 0.1 ms |
| 航迹管理 | O(n) | < 0.1 ms |
| **单周期总计** | | **< 10 ms** |

其中 n = 状态维度 = 6, m = 量测数, r = 模型数 = 3,
|Θ| = 分类类别数 = 4。

融合计算总耗时远小于 200 ms 周期，留有充分余量。

---

*文档结束*
