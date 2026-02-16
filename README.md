# 双目摄像头无人机检测仿真系统

Stereo Vision UAV Detection Simulation System

## 概述

本项目实现了一套装载于无人机平台的双目（立体）摄像头图像处理仿真系统，
用于检测、跟踪和定位高速运动的目标无人机（相对速度 >1000 km/h）。

## 主要功能

- **双目立体视觉**：立体校正、SGBM 视差计算、三角测量深度估计
- **目标检测**：模拟检测器（预留 YOLO 接口）、多尺度检测
- **多目标跟踪**：9 维 EKF（位置/速度/加速度）+ Hungarian 匹配
- **三维定位**：方位角、俯仰角、斜距、相对速度、碰撞预警时间
- **场景仿真**：8 类典型空战场景的合成与评估

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行单场景仿真
python sim/run_simulation.py --scenario s1_head_on

# 运行全部场景评估
python sim/batch_evaluation.py --all

# 生成仿真报告
python sim/report_generator.py --input results/ --output report/
```

## 项目结构

```
config/          - 系统配置文件
src/             - 核心算法源码
scenarios/       - 仿真场景定义
tests/           - 单元测试和集成测试
sim/             - 仿真运行脚本
docs/            - 工程文档（35篇）
```

## 文档体系

详见 `docs/` 目录，每个模块包含：方案文档、设计文档、仿真文档、使用手册。

## 技术栈

- Python 3.10+
- OpenCV 4.8+（立体视觉）
- NumPy（数值计算）
- SciPy（优化与统计）
- Matplotlib（可视化）

## 许可证

Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights reserved.
