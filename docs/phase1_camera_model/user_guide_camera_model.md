# 双目相机模型 — 使用手册

## 1. 快速入门

```python
from src.camera_model import StereoCamera

# 使用默认配置创建双目相机
stereo = StereoCamera(config_path="config/default_config.yaml")

# 将 3D 点投影到左右相机
point_3d = [10, -5, 200]  # X=10m, Y=-5m, Z=200m
uv_left = stereo.project_to_left(point_3d)
uv_right = stereo.project_to_right(point_3d)
print(f"Left pixel: {uv_left}, Right pixel: {uv_right}")

# 从视差计算深度
disparity = uv_left[0] - uv_right[0]
depth = stereo.depth_from_disparity(disparity)
print(f"Disparity: {disparity:.2f} px, Depth: {depth:.1f} m")

# 从像素 + 深度恢复 3D 坐标
point_recovered = stereo.pixel_to_3d(uv_left, depth)
print(f"Recovered 3D: {point_recovered}")
```

---

## 2. 安装与依赖

```bash
pip install numpy pyyaml
```

无需 OpenCV（本模块纯 NumPy 实现）。

---

## 3. 配置参数详解

配置文件 `config/default_config.yaml` 中 `camera` 节的参数：

### 3.1 图像参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_width` | int | 1920 | 图像宽度（像素） |
| `image_height` | int | 1080 | 图像高度（像素） |

### 3.2 相机内参

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `left.fx` | float | 1200.0 | 左相机 X 方向焦距（像素） |
| `left.fy` | float | 1200.0 | 左相机 Y 方向焦距（像素） |
| `left.cx` | float | 960.0 | 左相机主点 X 坐标（像素） |
| `left.cy` | float | 540.0 | 左相机主点 Y 坐标（像素） |
| `left.distortion` | list | [-0.1, 0.01, 0, 0, 0] | 畸变系数 [k1,k2,p1,p2,k3] |

右相机参数格式相同（`right.*`）。

### 3.3 双目外参

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stereo.baseline` | float | 0.3 | 基线距离（米） |
| `stereo.rotation` | list | [0,0,0] | 左→右旋转（Rodrigues向量） |
| `stereo.translation` | list | [0.3,0,0] | 右相机在左相机坐标系中的位置 |

### 3.4 相机-机体安装参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cam_to_body.rotation` | list | [0,0,0] | 安装旋转 [yaw,pitch,roll] (rad) |
| `cam_to_body.translation` | list | [0.1,0,-0.05] | 安装偏移 (m) |

---

## 4. API 参考

### 4.1 MonoCamera 类

#### 构造函数

```python
cam = MonoCamera(fx=1200, fy=1200, cx=960, cy=540,
                 dist=[-0.1, 0.01, 0, 0, 0],
                 width=1920, height=1080)
```

#### project(point_3d, apply_distortion=False)

将相机坐标系中的 3D 点投影为像素坐标。

- **输入**：`point_3d` — [X, Y, Z] 数组（米）
- **输出**：[u, v] 像素坐标，若点在相机后方返回 `None`
- **示例**：

```python
uv = cam.project([0, 0, 100])  # -> [960.0, 540.0]
```

#### back_project(pixel, depth, remove_distortion=False)

将像素坐标 + 深度反投影为 3D 点。

- **输入**：`pixel` — [u, v], `depth` — Z 值（米）
- **输出**：[X, Y, Z] 数组
- **示例**：

```python
p = cam.back_project([960, 540], 100)  # -> [0, 0, 100]
```

#### pixel_to_ray(pixel, remove_distortion=False)

将像素坐标转换为单位方向射线。

- **输入**：[u, v]
- **输出**：单位向量 [dx, dy, dz]

#### is_in_frame(pixel, margin=0)

检查像素是否在图像范围内。

### 4.2 StereoCamera 类

#### 构造函数

```python
stereo = StereoCamera(config_path="config/default_config.yaml")
# 或
stereo = StereoCamera(config=my_config_dict)
```

#### project_to_left / project_to_right

投影到左/右相机，输入为左相机坐标系中的 3D 点。

#### triangulate(pixel_left, pixel_right)

从立体像素对三角测量 3D 位置。

```python
p3d = stereo.triangulate([970, 540], [966.4, 540])
# -> 约 [0, 0, 100] （视差 3.6px → 深度 100m）
```

#### depth_from_disparity(disparity)

Z = f * B / d

```python
depth = stereo.depth_from_disparity(3.6)  # -> 100.0
```

#### depth_from_target_size(bbox_height_px, known_height_m)

从目标尺寸估计深度。

```python
depth = stereo.depth_from_target_size(48, 0.4)  # -> 10.0
```

---

## 5. 使用示例

### 5.1 基本用法：投影与反投影

```python
from src.camera_model import StereoCamera
import numpy as np

stereo = StereoCamera()

# 目标在 200m 外
target = np.array([10.0, -5.0, 200.0])

# 投影到双目
uv_l = stereo.project_to_left(target)
uv_r = stereo.project_to_right(target)
print(f"Left: ({uv_l[0]:.1f}, {uv_l[1]:.1f})")
print(f"Right: ({uv_r[0]:.1f}, {uv_r[1]:.1f})")
print(f"Disparity: {uv_l[0] - uv_r[0]:.2f} px")
```

### 5.2 高级用法：自定义相机参数

```python
config = {
  'camera': {
    'image_width': 1280,
    'image_height': 720,
    'left': {
      'fx': 800, 'fy': 800, 'cx': 640, 'cy': 360,
      'distortion': [0, 0, 0, 0, 0]
    },
    'right': {
      'fx': 800, 'fy': 800, 'cx': 640, 'cy': 360,
      'distortion': [0, 0, 0, 0, 0]
    },
    'stereo': {
      'baseline': 0.5,
      'rotation': [0, 0, 0],
      'translation': [0.5, 0, 0]
    },
    'cam_to_body': {
      'rotation': [0, 0, 0],
      'translation': [0, 0, 0]
    }
  }
}
stereo = StereoCamera(config=config)
```

---

## 6. 坐标系变换工具

```python
from src.utils import (
  camera_to_body, body_to_ned,
  euler_to_rotation_matrix
)

# 相机 -> 体坐标
R_c2b = euler_to_rotation_matrix(0, 0, 0)  # 无安装旋转
T_c2b = [0.1, 0, -0.05]
p_body = camera_to_body([10, -5, 200], R_c2b, T_c2b)

# 体坐标 -> NED
roll, pitch, yaw = 0.05, -0.1, 1.57  # IMU 姿态
p_ned = body_to_ned(p_body, roll, pitch, yaw)
print(f"Target NED position: {p_ned}")
```

---

## 7. 常见问题 (FAQ)

**Q: project() 返回 None 怎么办？**

A: 说明 3D 点在相机后方（Z ≤ 0），检查点的坐标系是否正确。

**Q: 如何提高远距离深度估计精度？**

A: 使用 `depth_from_target_size()` 方法，利用目标已知物理尺寸估计深度，
或者使用融合方法（Phase 3 中实现）。

**Q: 畸变参数从哪里获取？**

A: 实际系统中通过 OpenCV 棋盘格标定获得。仿真中使用配置文件中的预设值。

---

## 8. 故障排除

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| KeyError: 'camera' | 配置文件格式错误 | 检查 YAML 缩进和字段名 |
| 投影结果偏移 | cx/cy 不在图像中心 | 确认主点参数正确 |
| 三角测量误差大 | 视差过小（远距离） | 改用尺寸先验估计 |
| 畸变去除不完全 | 畸变系数过大 | 增加迭代次数 |
