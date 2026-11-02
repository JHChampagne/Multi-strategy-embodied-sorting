# Adapted from WorkCell Part C, Ahmed Sohail Butt (Apache-2.0).
"""Eye-to-hand extrinsic calibration with a sphere target.

Why a sphere: a partial view of a cube gives a surface centroid biased toward
the camera (~8 mm here), which would poison the extrinsics. A sphere's center
is recoverable WITHOUT bias from any surface patch via least-squares sphere
fitting — the reason industrial cells calibrate with precision spheres.

Procedure: the sphere is placed at K known world positions (in a real cell:
mounted on the flange and moved via FK, or measured placements). For each,
its center is estimated in the CAMERA frame from the depth image alone; the
rigid transform (R, t) best mapping camera-frame centers onto the known world
positions (Kabsch, SVD) IS the camera extrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .camera import Intrinsics, backproject
from .pose import kabsch_3d

SPHERE_R = 0.015

# 8 table placements + 3 elevated (sphere resting on a block) so the
# correspondences are not coplanar — improves out-of-plane conditioning.
DEFAULT_PLACEMENTS = [
    (-0.12, -0.30, SPHERE_R),
    (0.00, -0.30, SPHERE_R),
    (0.12, -0.30, SPHERE_R),
    (-0.12, -0.22, SPHERE_R),
    (0.12, -0.22, SPHERE_R),
    (-0.10, -0.17, SPHERE_R),
    (0.02, -0.17, SPHERE_R),
    (0.12, -0.17, SPHERE_R),
    (-0.06, -0.26, 0.024 + SPHERE_R),
    (0.06, -0.20, 0.024 + SPHERE_R),
    (0.00, -0.23, 0.024 + SPHERE_R),
]


def sphere_mask(rgb: np.ndarray) -> np.ndarray:
    """Magenta ratio test — distinct from the red block (low B) and blue tray (low R)."""
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    return (r > 90) & (b > 90) & (r > 1.3 * g) & (b > 1.3 * g)


def fit_sphere(points: np.ndarray, radius: float = SPHERE_R, gn_iters: int = 10) -> tuple[np.ndarray, float]:
    """Least-squares sphere center from a partial surface patch.

    Algebraic initialization (linear in center and c^2 - r^2), then
    Gauss-Newton refinement with the KNOWN radius. Returns (center, rmse).
    """
    # Algebraic: ||p||^2 = 2 p.c + (r^2 - ||c||^2)  ->  A [c; k] = b
    A = np.concatenate([2 * points, np.ones((len(points), 1))], axis=1)
    b = (points**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]

    for _ in range(gn_iters):
        diff = points - c
        dist = np.linalg.norm(diff, axis=1)
        resid = dist - radius
        J = -diff / dist[:, None]
        step, *_ = np.linalg.lstsq(J, -resid, rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-7:
            break
    dist = np.linalg.norm(points - c, axis=1)
    return c, float(np.sqrt(np.mean((dist - radius) ** 2)))


@dataclass
class Calibration:
    cam_pos: np.ndarray
    R_wc: np.ndarray  # world <- camera rotation
    residual_mm: float  # RMS of Kabsch correspondence residuals

    def save(self, path: str | Path) -> None:
        np.savez(path, cam_pos=self.cam_pos, R_wc=self.R_wc, residual_mm=self.residual_mm)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        z = np.load(path)
        return cls(cam_pos=z["cam_pos"], R_wc=z["R_wc"], residual_mm=float(z["residual_mm"]))


def calibrate(env, placements=DEFAULT_PLACEMENTS, camera: str = "rgbd", verbose: bool = True) -> Calibration:
    """Run the placement loop and solve extrinsics. Uses the simulator only to
    PLACE the sphere (the real-world analog: FK or measured positions) — the
    center estimates come from the camera alone."""
    K = Intrinsics.from_fovy(env.camera_fovy(camera), width=env.img_hw[1], height=env.img_hw[0])
    adr = env.model.jnt_qposadr[env.model.joint("calib_sphere_free").id]

    # Park everything else out of view; arm at rest (sensing pose)
    env.reset(
        options={"layout": {name: (0.5 + 0.04 * i, 0.3, 0.0) for i, name in enumerate(env._body_names)}}
    )
    key = env.model.key("rest")
    env.data.qpos[:6] = key.qpos[:6]
    env.data.ctrl[:6] = key.ctrl[:6]

    centers_cam, world = [], []
    for k, p in enumerate(placements):
        env.data.qpos[adr : adr + 3] = p
        env.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(env.model, env.data)
        obs = env.rgbd(camera)
        mask = sphere_mask(obs["rgb"]) & np.isfinite(obs["depth"])
        if mask.sum() < 50:
            raise RuntimeError(f"placement {k}: sphere not visible ({int(mask.sum())} px)")
        pts_cam, _ = backproject(obs["depth"], K, mask)
        c_cam, rmse = fit_sphere(pts_cam)
        centers_cam.append(c_cam)
        world.append(np.asarray(p, dtype=float))
        if verbose:
            print(f"  placement {k:02d}: {int(mask.sum()):>5}px  sphere-fit rmse {rmse * 1000:.2f}mm")

    # Park the sphere again
    env.data.qpos[adr : adr + 3] = [0.5, 0.3, SPHERE_R]
    mujoco.mj_forward(env.model, env.data)

    P, Q = np.array(centers_cam), np.array(world)
    R, t = kabsch_3d(P, Q)
    resid = np.linalg.norm(P @ R.T + t - Q, axis=1)
    return Calibration(cam_pos=t, R_wc=R, residual_mm=float(np.sqrt(np.mean(resid**2)) * 1000))
