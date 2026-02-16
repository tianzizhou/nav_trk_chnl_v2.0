# =============================================================================
# Copyright Notice:
# Copyright (c) 2023-2026 Shanghai Wuben Technology Co., Ltd. All rights
# reserved. No part of this code may be reproduced, modified, or distributed
# without the written permission of Shanghai Wuben Technology Co., Ltd.
# Project Name : T2031
# Module Name  : camera_model
# Version      : 1.0.0
# Author       : UAV Vision Team
# Date         : 2026-02-16
#
# Version History:
# 1.0.0 - 2026-02-16: Initial version.
#
# Features include:
# 1. Pinhole camera model with Brown-Conrady distortion
# 2. Stereo camera geometry (baseline, rectification)
# 3. 3D-to-pixel projection and pixel-to-ray back-projection
#
# Notes:
# 1. This code is for internal use only.
# =============================================================================

import numpy as np
from src.utils import (
  euler_to_rotation_matrix,
  rodrigues_to_rotation_matrix,
  load_config,
)


class MonoCamera:
  """Single (monocular) pinhole camera model with distortion.

  Attributes:
    fx, fy:    Focal lengths in pixels.
    cx, cy:    Principal point in pixels.
    K:         3x3 intrinsic matrix.
    dist:      Distortion coefficients [k1, k2, p1, p2, k3].
    width:     Image width in pixels.
    height:    Image height in pixels.
  """

  def __init__(self, fx, fy, cx, cy, dist=None,
               width=1920, height=1080):
    """Initialize monocular camera.

    Args:
      fx:     Focal length X (pixels).
      fy:     Focal length Y (pixels).
      cx:     Principal point X (pixels).
      cy:     Principal point Y (pixels).
      dist:   Distortion coefficients [k1, k2, p1, p2, k3].
      width:  Image width (pixels).
      height: Image height (pixels).
    """
    self.fx = float(fx)
    self.fy = float(fy)
    self.cx = float(cx)
    self.cy = float(cy)
    self.width = int(width)
    self.height = int(height)

    self.K = np.array([
      [self.fx, 0.0, self.cx],
      [0.0, self.fy, self.cy],
      [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    if dist is not None:
      self.dist = np.array(dist, dtype=np.float64).flatten()
    else:
      self.dist = np.zeros(5, dtype=np.float64)

  def project(self, point_3d, apply_distortion=False):
    """Project a 3D point (camera frame) to pixel coordinates.

    Args:
      point_3d:          [X, Y, Z] in camera frame (meters).
      apply_distortion:  Whether to apply lens distortion.

    Returns:
      np.ndarray: [u, v] pixel coordinates, or None if behind camera.
    """
    p = np.asarray(point_3d, dtype=np.float64).flatten()

    if p[2] <= 0:
      return None

    # Normalized coordinates
    x_n = p[0] / p[2]
    y_n = p[1] / p[2]

    if apply_distortion:
      x_n, y_n = self._apply_distortion(x_n, y_n)

    u = self.fx * x_n + self.cx
    v = self.fy * y_n + self.cy

    return np.array([u, v])

  def back_project(self, pixel, depth,
                   remove_distortion=False):
    """Back-project pixel coordinates to 3D point in camera frame.

    Args:
      pixel:              [u, v] pixel coordinates.
      depth:              Depth Z in meters.
      remove_distortion:  Whether to remove lens distortion first.

    Returns:
      np.ndarray: [X, Y, Z] in camera frame (meters).
    """
    pixel = np.asarray(pixel, dtype=np.float64).flatten()
    u, v = pixel[0], pixel[1]

    # Normalized coordinates
    x_n = (u - self.cx) / self.fx
    y_n = (v - self.cy) / self.fy

    if remove_distortion:
      x_n, y_n = self._remove_distortion(x_n, y_n)

    X = x_n * depth
    Y = y_n * depth
    Z = depth

    return np.array([X, Y, Z])

  def pixel_to_ray(self, pixel, remove_distortion=False):
    """Convert pixel coordinates to a unit ray in camera frame.

    Args:
      pixel:              [u, v] pixel coordinates.
      remove_distortion:  Whether to remove distortion.

    Returns:
      np.ndarray: Unit direction vector [dx, dy, dz].
    """
    pixel = np.asarray(pixel, dtype=np.float64).flatten()

    x_n = (pixel[0] - self.cx) / self.fx
    y_n = (pixel[1] - self.cy) / self.fy

    if remove_distortion:
      x_n, y_n = self._remove_distortion(x_n, y_n)

    ray = np.array([x_n, y_n, 1.0])
    return ray / np.linalg.norm(ray)

  def is_in_frame(self, pixel, margin=0):
    """Check if pixel coordinates are within the image frame.

    Args:
      pixel:  [u, v] pixel coordinates.
      margin: Pixel margin from edges.

    Returns:
      bool: True if pixel is inside the image.
    """
    if pixel is None:
      return False
    u, v = pixel[0], pixel[1]
    return (margin <= u < self.width - margin
            and margin <= v < self.height - margin)

  def _apply_distortion(self, x_n, y_n):
    """Apply Brown-Conrady distortion to normalized coordinates.

    Args:
      x_n, y_n: Normalized (undistorted) image coordinates.

    Returns:
      tuple: (x_d, y_d) distorted normalized coordinates.
    """
    k1, k2, p1, p2, k3 = self.dist[:5]

    r2 = x_n * x_n + y_n * y_n
    r4 = r2 * r2
    r6 = r4 * r2

    # Radial distortion
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    # Tangential distortion
    x_d = x_n * radial + 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n * x_n)
    y_d = y_n * radial + p1 * (r2 + 2 * y_n * y_n) + 2 * p2 * x_n * y_n

    return x_d, y_d

  def _remove_distortion(self, x_d, y_d, iterations=10):
    """Remove distortion from normalized coordinates iteratively.

    Uses fixed-point iteration to invert the distortion model.

    Args:
      x_d, y_d:   Distorted normalized coordinates.
      iterations: Number of iterations.

    Returns:
      tuple: (x_n, y_n) undistorted normalized coordinates.
    """
    x_n = x_d
    y_n = y_d

    for _ in range(iterations):
      x_est, y_est = self._apply_distortion(x_n, y_n)
      x_n = x_n + (x_d - x_est)
      y_n = y_n + (y_d - y_est)

    return x_n, y_n


class StereoCamera:
  """Stereo camera model with left-right geometry.

  Attributes:
    left:      MonoCamera for the left camera.
    right:     MonoCamera for the right camera.
    baseline:  Baseline distance in meters.
    R:         3x3 rotation from left to right camera.
    T:         3D translation from left to right camera.
    R_cam2body: 3x3 rotation from camera to body frame.
    T_cam2body: 3D translation from camera to body frame.
  """

  def __init__(self, config=None, config_path=None):
    """Initialize stereo camera from configuration.

    Args:
      config:      Configuration dictionary. If None, loaded from file.
      config_path: Path to config file (used if config is None).
    """
    if config is None:
      if config_path is None:
        config_path = "config/default_config.yaml"
      config = load_config(config_path)

    cam_cfg = config['camera']
    width = cam_cfg['image_width']
    height = cam_cfg['image_height']

    # Left camera
    left_cfg = cam_cfg['left']
    self.left = MonoCamera(
      fx=left_cfg['fx'], fy=left_cfg['fy'],
      cx=left_cfg['cx'], cy=left_cfg['cy'],
      dist=left_cfg.get('distortion'),
      width=width, height=height
    )

    # Right camera
    right_cfg = cam_cfg['right']
    self.right = MonoCamera(
      fx=right_cfg['fx'], fy=right_cfg['fy'],
      cx=right_cfg['cx'], cy=right_cfg['cy'],
      dist=right_cfg.get('distortion'),
      width=width, height=height
    )

    # Stereo extrinsics
    stereo_cfg = cam_cfg['stereo']
    self.baseline = float(stereo_cfg['baseline'])
    self.R = rodrigues_to_rotation_matrix(
      stereo_cfg.get('rotation', [0, 0, 0])
    )
    self.T = np.array(
      stereo_cfg['translation'], dtype=np.float64
    )

    # Camera-to-body transformation
    # Camera: X-right, Y-down, Z-forward
    # Body:   X-forward, Y-right, Z-down
    c2b_cfg = cam_cfg.get('cam_to_body', {})

    if c2b_cfg.get('use_standard_transform', False):
      # Standard permutation: cam->body
      # cam_X(right)   -> body_Y(right)
      # cam_Y(down)    -> body_Z(down)
      # cam_Z(forward) -> body_X(forward)
      self.R_cam2body = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
      ], dtype=np.float64)
    else:
      c2b_rot = c2b_cfg.get('rotation', [0, 0, 0])
      self.R_cam2body = euler_to_rotation_matrix(
        c2b_rot[2], c2b_rot[1], c2b_rot[0]
      )

    self.T_cam2body = np.array(
      c2b_cfg.get('translation', [0, 0, 0]),
      dtype=np.float64
    )

  def project_to_left(self, point_3d, apply_distortion=False):
    """Project 3D point to left camera pixel coordinates.

    Args:
      point_3d:          [X, Y, Z] in left camera frame.
      apply_distortion:  Whether to apply lens distortion.

    Returns:
      np.ndarray or None: [u, v] pixel coordinates.
    """
    return self.left.project(point_3d, apply_distortion)

  def project_to_right(self, point_3d, apply_distortion=False):
    """Project 3D point to right camera pixel coordinates.

    The point is first transformed from left camera frame to
    right camera frame. Convention: T is the position of the
    right camera origin expressed in the left camera frame.
    So p_right = R * (p_left - T).

    Args:
      point_3d:          [X, Y, Z] in left camera frame.
      apply_distortion:  Whether to apply lens distortion.

    Returns:
      np.ndarray or None: [u, v] pixel coordinates.
    """
    p = np.asarray(point_3d, dtype=np.float64).flatten()
    # Transform from left to right camera frame
    # T is right camera position in left frame
    p_right = self.R @ (p - self.T)
    return self.right.project(p_right, apply_distortion)

  def triangulate(self, pixel_left, pixel_right):
    """Triangulate 3D position from stereo pixel correspondences.

    Uses disparity-based depth for the standard rectified case,
    and falls back to the midpoint method for the general case.

    Args:
      pixel_left:  [u, v] in left image.
      pixel_right: [u, v] in right image.

    Returns:
      np.ndarray: [X, Y, Z] in left camera frame (meters).
    """
    pixel_left = np.asarray(pixel_left, dtype=np.float64)
    pixel_right = np.asarray(pixel_right, dtype=np.float64)

    # For standard rectified stereo (R ~ I), use simple disparity
    is_rectified = np.allclose(self.R, np.eye(3), atol=1e-6)

    if is_rectified:
      return self._depth_from_disparity_pixels(
        pixel_left, pixel_right
      )

    # General case: midpoint triangulation
    # Left ray: origin at [0,0,0], direction = ray_l
    ray_l = self.left.pixel_to_ray(pixel_left)

    # Right ray: origin at T (in left frame),
    #   direction from right camera transformed to left frame
    ray_r_cam = self.right.pixel_to_ray(pixel_right)
    ray_r = self.R.T @ ray_r_cam

    origin_r = self.T.copy()

    # Closest point on two rays:
    # P_l = s * ray_l
    # P_r = origin_r + t * ray_r
    w0 = -origin_r  # origin_l(=0) - origin_r
    a = ray_l @ ray_l
    b = ray_l @ ray_r
    c = ray_r @ ray_r
    d = ray_l @ w0
    e = ray_r @ w0

    denom = a * c - b * b
    if abs(denom) < 1e-12:
      return self._depth_from_disparity_pixels(
        pixel_left, pixel_right
      )

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    p_l = s * ray_l
    p_r = origin_r + t * ray_r

    return (p_l + p_r) / 2.0

  def depth_from_disparity(self, disparity):
    """Compute depth from disparity value.

    Z = f * B / d

    Args:
      disparity: Disparity in pixels (must be > 0).

    Returns:
      float: Depth in meters, or None if disparity invalid.
    """
    if disparity <= 0:
      return None
    return self.left.fx * self.baseline / disparity

  def disparity_from_depth(self, depth):
    """Compute expected disparity from depth.

    d = f * B / Z

    Args:
      depth: Depth in meters (must be > 0).

    Returns:
      float: Disparity in pixels.
    """
    if depth <= 0:
      return 0.0
    return self.left.fx * self.baseline / depth

  def depth_from_target_size(self, bbox_height_pixels,
                             known_height_meters):
    """Estimate depth from target bounding box size and known size.

    Z = known_height * f / bbox_height

    Args:
      bbox_height_pixels:  Target height in pixels.
      known_height_meters: Known physical height in meters.

    Returns:
      float: Estimated depth in meters, or None if invalid.
    """
    if bbox_height_pixels <= 0:
      return None
    return known_height_meters * self.left.fy / bbox_height_pixels

  def pixel_to_3d(self, pixel_left, depth):
    """Convert left camera pixel + depth to 3D position.

    Args:
      pixel_left: [u, v] in left image.
      depth:      Depth Z in meters.

    Returns:
      np.ndarray: [X, Y, Z] in left camera frame.
    """
    return self.left.back_project(pixel_left, depth)

  def _depth_from_disparity_pixels(self, pixel_left, pixel_right):
    """Fallback depth computation using horizontal disparity.

    Args:
      pixel_left:  [u, v] in left image.
      pixel_right: [u, v] in right image.

    Returns:
      np.ndarray: [X, Y, Z] in left camera frame.
    """
    disp = pixel_left[0] - pixel_right[0]
    depth = self.depth_from_disparity(disp)
    if depth is None:
      return np.array([0, 0, 0], dtype=np.float64)
    return self.left.back_project(pixel_left, depth)
