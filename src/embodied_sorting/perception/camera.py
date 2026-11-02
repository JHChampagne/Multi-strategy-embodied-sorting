# Adapted from WorkCell Part C, Ahmed Sohail Butt (Apache-2.0).
"""Pinhole camera model for the RGB-D sensor: intrinsics + back-projection.

MuJoCo camera conventions (verified empirically in scripts/verify_camera.py):
the camera looks along its local -Z axis with +X right and +Y up (OpenGL),
`cam_fovy` is the vertical field of view in degrees, and the depth renderer
returns metric z-depth (distance along the viewing axis, not ray length).

Image row v increases downward while camera +Y points up, hence the sign flip
on the y term of the back-projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int) -> "Intrinsics":
        fy = height / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
        return cls(fx=fy, fy=fy, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0, width=width, height=height)

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def backproject(
    depth: np.ndarray, K: Intrinsics, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Depth image -> 3D points in the CAMERA (OpenGL) frame.

    Args:
        depth: (H, W) metric z-depth.
        mask: optional (H, W) bool; only these pixels are back-projected.

    Returns:
        (points, pixels): (N, 3) camera-frame points and the (N, 2) integer
        (u, v) pixel coordinates they came from.
    """
    h, w = depth.shape
    if mask is None:
        mask = np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(mask)
    d = depth[v, u]
    x = (u - K.cx) / K.fx * d
    y = -(v - K.cy) / K.fy * d  # image v grows down; camera +Y is up
    z = -d  # camera looks along -Z
    return np.stack([x, y, z], axis=1), np.stack([u, v], axis=1)


def cam_to_world(points_cam: np.ndarray, cam_pos: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
    """Camera-frame points -> world frame given extrinsics (R_wc: world<-cam)."""
    return points_cam @ R_wc.T + cam_pos


def world_to_cam(points_w: np.ndarray, cam_pos: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
    return (points_w - cam_pos) @ R_wc


def project(points_cam: np.ndarray, K: Intrinsics) -> np.ndarray:
    """Camera-frame points -> (u, v) pixels (float). Points must have z < 0."""
    z = -points_cam[:, 2]
    u = points_cam[:, 0] / z * K.fx + K.cx
    v = -points_cam[:, 1] / z * K.fy + K.cy
    return np.stack([u, v], axis=1)
