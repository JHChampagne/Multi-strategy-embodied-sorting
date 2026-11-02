# Adapted from WorkCell Part C, Ahmed Sohail Butt (Apache-2.0).
"""Point-cloud processing, implemented from scratch (NumPy + scipy cKDTree).

No Open3D. RANSAC plane segmentation, voxel-grid downsampling, and DBSCAN
Euclidean clustering are all first-principles implementations — the point of
this project is demonstrating the geometry, not calling a library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# ------------------------------------------------------------------ downsample
def voxel_downsample(
    points: np.ndarray, colors: np.ndarray | None, voxel: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Average points (and colors) within each `voxel`-sized grid cell."""
    keys = np.floor(points / voxel).astype(np.int64)
    # Unique voxel per point -> group by inverse index
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n_vox = counts.shape[0]
    out = np.zeros((n_vox, 3))
    np.add.at(out, inv, points)
    out /= counts[:, None]
    out_c = None
    if colors is not None:
        out_c = np.zeros((n_vox, colors.shape[1]))
        np.add.at(out_c, inv, colors.astype(float))
        out_c /= counts[:, None]
    return out, out_c


# ----------------------------------------------------------------- plane RANSAC
@dataclass
class Plane:
    normal: np.ndarray  # unit normal
    d: float  # plane: normal . p + d = 0

    def distance(self, points: np.ndarray) -> np.ndarray:
        return np.abs(points @ self.normal + self.d)


def ransac_plane(
    points: np.ndarray,
    threshold: float = 0.003,
    n_iters: int = 250,
    rng: np.random.Generator | None = None,
) -> tuple[Plane, np.ndarray]:
    """Largest-plane fit by RANSAC + least-squares refinement.

    Returns (plane, inlier_mask).
    """
    rng = rng or np.random.default_rng(0)
    n = len(points)
    best_count, best_inliers = -1, None
    for _ in range(n_iters):
        i, j, k = rng.choice(n, size=3, replace=False)
        v1, v2 = points[j] - points[i], points[k] - points[i]
        nrm = np.cross(v1, v2)
        norm = np.linalg.norm(nrm)
        if norm < 1e-12:
            continue
        nrm = nrm / norm
        d = -nrm @ points[i]
        inliers = np.abs(points @ nrm + d) < threshold
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_inliers = c, inliers

    # Least-squares refinement on the consensus set (SVD of centered points)
    p = points[best_inliers]
    centroid = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - centroid, full_matrices=False)
    normal = vt[2]
    if normal[2] < 0:  # orient +z (table up)
        normal = -normal
    plane = Plane(normal=normal, d=float(-normal @ centroid))
    return plane, plane.distance(points) < threshold


# --------------------------------------------------------------------- DBSCAN
def dbscan(points: np.ndarray, eps: float = 0.009, min_samples: int = 8) -> np.ndarray:
    """Euclidean clustering. Returns labels (N,), -1 = noise.

    Core points have >= min_samples neighbors within eps; clusters grow by
    breadth-first expansion through core points; border points join the first
    cluster that reaches them.
    """
    n = len(points)
    tree = cKDTree(points)
    neighborhoods = tree.query_ball_point(points, eps)
    core = np.fromiter((len(nb) >= min_samples for nb in neighborhoods), dtype=bool, count=n)

    labels = np.full(n, -1, dtype=int)
    cluster = 0
    for seed in range(n):
        if labels[seed] != -1 or not core[seed]:
            continue
        labels[seed] = cluster
        stack = [seed]
        while stack:
            j = stack.pop()
            for k in neighborhoods[j]:
                if labels[k] == -1:
                    labels[k] = cluster
                    if core[k]:
                        stack.append(k)
        cluster += 1
    return labels


# ------------------------------------------------------------ object extraction
@dataclass
class Cluster:
    points: np.ndarray
    colors: np.ndarray
    centroid: np.ndarray


def extract_objects(
    points_w: np.ndarray,
    colors: np.ndarray,
    workspace: tuple[tuple[float, float], ...] = ((-0.25, 0.25), (-0.33, -0.14), (0.004, 0.06)),
    voxel: float = 0.003,
    min_cluster_points: int = 25,
    rng: np.random.Generator | None = None,
) -> tuple[list[Cluster], Plane]:
    """Full tabletop pipeline: plane removal -> ROI crop -> clustering.

    Returns object clusters (largest first) and the fitted table plane.
    """
    pts, cols = voxel_downsample(points_w, colors, voxel)

    plane, plane_mask = ransac_plane(pts, rng=rng)
    off_plane = ~plane_mask

    roi = np.ones(len(pts), dtype=bool)
    for axis, (lo, hi) in enumerate(workspace):
        roi &= (pts[:, axis] >= lo) & (pts[:, axis] <= hi)
    keep = off_plane & roi
    pts, cols = pts[keep], cols[keep]
    if len(pts) == 0:
        return [], plane

    labels = dbscan(pts)
    clusters = []
    for cid in range(labels.max() + 1):
        m = labels == cid
        if int(m.sum()) < min_cluster_points:
            continue
        clusters.append(Cluster(points=pts[m], colors=cols[m], centroid=pts[m].mean(axis=0)))
    clusters.sort(key=lambda c: -len(c.points))
    return clusters, plane


def select_target(clusters: list[Cluster]) -> Cluster | None:
    """Pick the red cluster (dominant-red ratio test on mean color)."""
    best, best_score = None, 0.0
    for c in clusters:
        r, g, b = c.colors.mean(axis=0)[:3]
        if r > 90 and r > 1.4 * g and r > 1.4 * b:
            score = r / (g + b + 1e-6)
            if score > best_score:
                best, best_score = c, score
    return best
