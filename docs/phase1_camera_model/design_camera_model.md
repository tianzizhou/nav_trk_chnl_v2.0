# 双目相机模型 — 详细设计文档

## 1. 模块架构

```
src/camera_model.py          src/utils.py
┌─────────────────┐          ┌────────────────────────┐
│ MonoCamera       │          │ Rotation Conversions   │
│ ├─ project()     │          │ ├─ euler_to_rotation() │
│ ├─ back_project()│◄────────│ ├─ rotation_to_euler() │
│ ├─ pixel_to_ray()│          │ ├─ euler_to_quat()     │
│ ├─ is_in_frame() │          │ ├─ quat_to_euler()     │
│ └─ distortion    │          │ └─ rodrigues_to_mat()  │
├─────────────────┤          ├────────────────────────┤
│ StereoCamera     │          │ Coord Transforms       │
│ ├─ project_L/R() │          │ ├─ camera_to_body()    │
│ ├─ triangulate() │          │ ├─ body_to_camera()    │
│ ├─ depth_disp()  │          │ ├─ body_to_ned()       │
│ └─ depth_size()  │          │ └─ ned_to_body()       │
└─────────────────┘          ├────────────────────────┤
                              │ Math Utils             │
                              │ ├─ normalize_angle()   │
                              │ ├─ unit_vector()       │
                              │ └─ skew_symmetric()    │
                              ├────────────────────────┤
                              │ Config                 │
                              │ └─ load_config()       │
                              └────────────────────────┘
```

---

## 2. 类/函数接口定义

### 2.1 MonoCamera 类

```python
class MonoCamera:
  def __init__(self, fx, fy, cx, cy,
               dist=None, width=1920, height=1080):
    """
    Args:
      fx, fy:  Focal lengths (pixels).
      cx, cy:  Principal point (pixels).
      dist:    [k1, k2, p1, p2, k3] distortion coefficients.
      width:   Image width (pixels).
      height:  Image height (pixels).

    Attributes:
      K:    3x3 intrinsic matrix (np.ndarray).
      dist: 5-element distortion vector (np.ndarray).
    """

  def project(self, point_3d, apply_distortion=False):
    """Project 3D point (camera frame) to pixel [u, v].
    Returns None if point is behind camera (Z <= 0)."""

  def back_project(self, pixel, depth,
                   remove_distortion=False):
    """Back-project [u, v] + depth Z to [X, Y, Z]."""

  def pixel_to_ray(self, pixel, remove_distortion=False):
    """Convert [u, v] to unit direction ray [dx, dy, dz]."""

  def is_in_frame(self, pixel, margin=0):
    """Check if [u, v] is within image bounds."""
```

### 2.2 StereoCamera 类

```python
class StereoCamera:
  def __init__(self, config=None, config_path=None):
    """
    Attributes:
      left:       MonoCamera (left camera).
      right:      MonoCamera (right camera).
      baseline:   Baseline distance (meters).
      R:          3x3 rotation (left -> right camera).
      T:          3D position of right cam in left frame.
      R_cam2body: 3x3 rotation (camera -> body frame).
      T_cam2body: 3D translation (camera -> body frame).
    """

  def project_to_left(self, point_3d, apply_distortion=False):
    """Project to left camera pixels."""

  def project_to_right(self, point_3d, apply_distortion=False):
    """Project to right camera pixels.
    p_right = R * (p_left - T)"""

  def triangulate(self, pixel_left, pixel_right):
    """Stereo triangulation -> [X, Y, Z] in left frame."""

  def depth_from_disparity(self, disparity):
    """Z = f * B / d. Returns None if d <= 0."""

  def disparity_from_depth(self, depth):
    """d = f * B / Z."""

  def depth_from_target_size(self, bbox_height_px,
                             known_height_m):
    """Z = known_height * fy / bbox_height."""

  def pixel_to_3d(self, pixel_left, depth):
    """[u, v] + Z -> [X, Y, Z] in left camera frame."""
```

### 2.3 utils.py 函数

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `euler_to_rotation_matrix(r,p,y)` | 3 floats (rad) | 3x3 ndarray | ZYX 惯例 |
| `rotation_matrix_to_euler(R)` | 3x3 ndarray | (r,p,y) tuple | |
| `euler_to_quaternion(r,p,y)` | 3 floats | [w,x,y,z] | |
| `quaternion_to_euler(q)` | [w,x,y,z] | (r,p,y) | |
| `quaternion_to_rotation_matrix(q)` | [w,x,y,z] | 3x3 ndarray | |
| `rotation_matrix_to_quaternion(R)` | 3x3 ndarray | [w,x,y,z] | |
| `rodrigues_to_rotation_matrix(rvec)` | (3,) vector | 3x3 ndarray | |
| `camera_to_body(p, R, T)` | point, R, T | point | |
| `body_to_camera(p, R, T)` | point, R, T | point | inverse |
| `body_to_ned(p, r, p, y)` | point, angles | point | |
| `ned_to_body(p, r, p, y)` | point, angles | point | inverse |
| `normalize_angle(a)` | float (rad) | float | → [-π, π] |
| `unit_vector(v)` | ndarray | ndarray | ‖v‖ = 1 |
| `skew_symmetric(v)` | (3,) | 3x3 | [v]× |
| `load_config(path)` | str | dict | YAML 加载 |

---

## 3. 核心算法

### 3.1 针孔投影模型

**正向投影（3D → 像素）**：

给定相机坐标系中的 3D 点 P = (X, Y, Z)：

1. 归一化坐标：x_n = X/Z, y_n = Y/Z
2. 畸变（可选）：(x_d, y_d) = distort(x_n, y_n)
3. 像素坐标：u = fx * x_d + cx, v = fy * y_d + cy

**反向投影（像素 → 3D）**：

给定像素 (u, v) 和深度 Z：

1. 归一化坐标：x_n = (u - cx) / fx, y_n = (v - cy) / fy
2. 去畸变（可选）：(x_u, y_u) = undistort(x_n, y_n)
3. 3D 坐标：X = x_u * Z, Y = y_u * Z

### 3.2 Brown-Conrady 畸变模型

畸变参数：k1, k2, p1, p2, k3

正向畸变（undistorted → distorted）：

```
r² = x² + y²
radial = 1 + k1*r² + k2*r⁴ + k3*r⁶
x_d = x * radial + 2*p1*x*y + p2*(r² + 2*x²)
y_d = y * radial + p1*(r² + 2*y²) + 2*p2*x*y
```

逆向去畸变使用迭代法（fixed-point iteration, 10 次迭代）。

### 3.3 立体三角测量

**标准矫正情况**（R ≈ I）：

使用简单视差公式：d = u_left - u_right, Z = f*B/d

**一般情况**：

使用最近点法（midpoint method），求两条射线的最近点对。

### 3.4 Euler 角 (ZYX) 旋转矩阵

旋转顺序：先 yaw (Z轴), 再 pitch (Y轴), 最后 roll (X轴)。

R = Rz(ψ) · Ry(θ) · Rx(φ)

---

## 4. 配置参数

所有相机参数从 `config/default_config.yaml` 的 `camera` 节加载：

| 参数路径 | 默认值 | 说明 |
|----------|--------|------|
| `camera.image_width` | 1920 | 图像宽度 |
| `camera.image_height` | 1080 | 图像高度 |
| `camera.left.fx` | 1200.0 | 左相机焦距 X |
| `camera.left.fy` | 1200.0 | 左相机焦距 Y |
| `camera.left.cx` | 960.0 | 主点 X |
| `camera.left.cy` | 540.0 | 主点 Y |
| `camera.left.distortion` | [-0.1, 0.01, 0, 0, 0] | 畸变系数 |
| `camera.stereo.baseline` | 0.3 | 基线距离 (m) |
| `camera.stereo.translation` | [0.3, 0, 0] | 右相机位置 |
| `camera.cam_to_body.rotation` | [0, 0, 0] | 安装姿态 |
| `camera.cam_to_body.translation` | [0.1, 0, -0.05] | 安装偏移 |

---

## 5. 错误处理

| 条件 | 处理方式 |
|------|----------|
| Z ≤ 0 (点在相机后方) | `project()` 返回 None |
| 视差 ≤ 0 | `depth_from_disparity()` 返回 None |
| bbox_height ≤ 0 | `depth_from_target_size()` 返回 None |
| 配置文件缺失 | 抛出 FileNotFoundError |
| 参数格式错误 | 抛出 KeyError / ValueError |

---

## 6. 性能考量

- 所有矩阵运算使用 NumPy，向量化高效
- 畸变逆模型使用 10 次迭代（精度 < 0.01 px）
- 单次投影/反投影耗时 < 0.01 ms

---

## 7. 依赖关系

- `numpy` ≥ 1.24
- `pyyaml` ≥ 6.0（配置加载）
- 无外部 OpenCV 依赖（本模块纯 NumPy 实现）
