# Adapted from WorkCell Part C, Ahmed Sohail Butt (Apache-2.0).
"""6-DoF pose estimation for a tabletop cube: PCA init + table-constrained ICP.

The physics does half the work: a cube resting on a known table plane has
z = plane + half, roll = pitch = 0 — so the free parameters are (x, y, yaw).
ICP therefore runs in a constrained 3-DoF form: 3D nearest-neighbor
correspondences against a model of the cube's camera-visible surface (top +
4 sides, no bottom), with a closed-form 2D Kabsch solve per iteration.

Yaw is identifiable only mod 90 deg (square symmetry) — downstream grasp
planning treats the four face pairs as equivalent candidates.

All from scratch: Kabsch via SVD, ICP loop with scipy cKDTree matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# ------------------------------------------------------------------- Kabsch
def kabsch_2d(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit 2D rotation R and translation t with R @ p + t ~= q (SVD)."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = cq - R @ cp
    return R, t


def kabsch_3d(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit 3D rigid transform (R, t) with R @ p + t ~= q. Used by the
    eye-to-hand calibration."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    return R, t


# ---------------------------------------------------------------- cube model
def cube_surface_model(half: float, grid: float = 0.0025) -> np.ndarray:
    """Points sampled on the cube's camera-visible surface (top + 4 sides),
    in the cube's local frame (origin at center)."""
    ax = np.arange(-half, half + 1e-9, grid)
    u, v = np.meshgrid(ax, ax, indexing="ij")
    u, v = u.ravel(), v.ravel()
    h = np.full_like(u, half)
    faces = [
        np.stack([u, v, h], axis=1),  # top (+z)
        np.stack([u, h, v], axis=1),  # +y side
        np.stack([u, -h, v], axis=1),  # -y side
        np.stack([h, u, v], axis=1),  # +x side
        np.stack([-h, u, v], axis=1),  # -x side
    ]
    return np.concatenate(faces, axis=0)


@dataclass
class CubePose:
    position: np.ndarray  # world xyz of the cube center
    yaw: float  # radians, mod pi/2
    rmse: float  # final ICP residual [m]
    n_iters: int


def estimate_cube_pose(
    cluster_points: np.ndarray,
    table_z: float,
    half: float = 0.012,
    max_iters: int = 40,
    tol: float = 1e-5,
) -> CubePose:
    """Table-constrained ICP fit of a cube to an observed cluster.

    Args:
        cluster_points: (N, 3) world-frame points of the segmented object.
        table_z: z of the fitted table plane (from RANSAC).
    """
    model = cube_surface_model(half)
    z_center = table_z + half

    # ---- init: global coarse search over the symmetry-reduced yaw range.
    # PCA is useless here (a square's xy projection has near-equal
    # eigenvalues, so the principal axis is noise) and a bad init drops ICP
    # into the 45-deg corner-facing local minimum. Yaw only lives in
    # [0, 90) deg, so brute-force it coarsely and let ICP polish.
    xy = cluster_points[:, :2]
    t_xy = xy.mean(axis=0).copy()
    obs_tree = cKDTree(cluster_points)
    best_yaw, best_cost = 0.0, np.inf
    for cand in np.deg2rad(np.arange(0.0, 90.0, 5.0)):
        cy, sy = np.cos(cand), np.sin(cand)
        Rz = np.array([[cy, -sy], [sy, cy]])
        mw = np.empty_like(model)
        mw[:, :2] = model[:, :2] @ Rz.T + t_xy
        mw[:, 2] = model[:, 2] + z_center
        d, _ = obs_tree.query(mw)
        # model->observed distances, trimmed: unobserved model faces (back
        # sides) shouldn't dominate the score
        cost = float(np.mean(np.sort(d)[: int(0.6 * len(d))]))
        if cost < best_cost:
            best_cost, best_yaw = cost, float(cand)
    yaw = best_yaw

    prev_rmse = np.inf
    n_done = 0
    for it in range(max_iters):
        n_done = it + 1
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy], [sy, cy]])
        # model -> world under current estimate
        mw = np.empty_like(model)
        mw[:, :2] = model[:, :2] @ Rz.T + t_xy
        mw[:, 2] = model[:, 2] + z_center

        # correspondences: each observed point -> nearest model point (3D)
        tree = cKDTree(mw)
        dist, idx = tree.query(cluster_points)
        rmse = float(np.sqrt(np.mean(dist**2)))

        # trim the worst 10% (occlusion boundaries, depth-edge artifacts)
        keep = dist <= np.quantile(dist, 0.9)
        src = mw[idx[keep], :2]  # matched model points (world xy)
        dst = cluster_points[keep, :2]  # observed xy

        # 2D Kabsch increment: rotate/translate the model onto the observations
        R_inc, t_inc = kabsch_2d(src, dst)
        yaw = (yaw + float(np.arctan2(R_inc[1, 0], R_inc[0, 0]))) % (np.pi / 2)
        t_xy = R_inc @ t_xy + t_inc

        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    return CubePose(
        position=np.array([t_xy[0], t_xy[1], z_center]),
        yaw=yaw,
        rmse=rmse,
        n_iters=n_done,
    )
