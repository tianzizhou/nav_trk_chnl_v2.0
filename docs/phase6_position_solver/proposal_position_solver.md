# 位置解算模块 — 技术方案文档

## 1. 概述

位置解算模块将跟踪器输出的相机坐标系位置/速度，转换为方位角、俯仰角、
斜距、相对速度和碰撞预警时间(TTC)。

## 2. 方位角定义

- **方位角(Azimuth)**：目标在相机XZ平面的偏角，以Z轴(正前方)为0°
  - 正值 = 右侧，负值 = 左侧
  - α = atan2(X_cam, Z_cam)

- **俯仰角(Elevation)**：目标在YZ平面的偏角
  - 正值 = 上方（相机Y轴向下），负值 = 下方
  - β = atan2(-Y_cam, Z_cam)

## 3. TTC计算方法

```
closing_speed = -vel_cam · (pos_cam / |pos_cam|)
TTC = |pos_cam| / closing_speed  (if closing_speed > 0)
```

## 4. 威胁评估

| TTC范围 | 等级 | 颜色 |
|---------|------|------|
| > 5s | safe | 绿色 |
| 2~5s | warning | 黄色 |
| < 2s | critical | 红色 |

## 5. 坐标系变换链

Camera → Body → NED（通过IMU姿态）
