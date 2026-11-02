"""Calibrated RGB-D geometry and color recognition, without simulator segmentation."""

from dataclasses import dataclass
import numpy as np

from .camera import Intrinsics, backproject, cam_to_world
from .pointcloud import voxel_downsample, ransac_plane, dbscan
from .pose import estimate_cube_pose


@dataclass
class Detection:
    color: str
    position: np.ndarray
    yaw: float
    rmse: float
    points: int


class PerceptionPipeline:
    def __init__(self, calibration, fovy, seed=0):
        if calibration is None:
            raise ValueError("Vision needs an offline sphere calibration; run workcell calibrate")
        self.calibration = calibration
        self.K = Intrinsics.from_fovy(fovy, width=640, height=480)
        self.rng = np.random.default_rng(seed)

    def detect(self, rgbd):
        pts, pixels = backproject(rgbd["depth"], self.K)
        pts = cam_to_world(pts, self.calibration.cam_pos, self.calibration.R_wc)
        cols = rgbd["rgb"][pixels[:, 1], pixels[:, 0]]
        pts, cols = voxel_downsample(pts, cols, 0.002)
        if len(pts) < 100:
            raise RuntimeError("Insufficient valid depth")
        sample = pts[self.rng.choice(len(pts), min(6000, len(pts)), replace=False)]
        plane, _ = ransac_plane(sample, n_iters=100, rng=self.rng)
        if plane.normal[2] < 0.95:
            raise RuntimeError("Table calibration inconsistent with sensed plane")
        table_z = -plane.d / plane.normal[2]
        roi = (
            (pts[:, 0] > -0.30)
            & (pts[:, 0] < 0.30)
            & (pts[:, 1] > -0.38)
            & (pts[:, 1] < 0.08)
            & (pts[:, 2] > table_z + 0.004)
            & (pts[:, 2] < table_z + 0.060)
        )
        r, g, b = cols.T
        masks = {
            "red": (r > 80) & (r > 1.4 * g) & (r > 1.4 * b),
            "green": (g > 65) & (g > 1.3 * r) & (g > 1.3 * b),
            "yellow": (r > 80) & (g > 80) & (r > 1.4 * b) & (g > 1.4 * b) & (r < 1.8 * g) & (g < 1.8 * r),
        }
        detections = {}
        for color, mask in masks.items():
            p = pts[roi & mask]
            if len(p) < 25:
                continue
            labels = dbscan(p, eps=0.008, min_samples=5)
            clusters = [p[labels == i] for i in range(labels.max() + 1)]
            if not clusters:
                continue
            p = max(clusters, key=len)
            if len(p) < 25:
                continue
            # The top face establishes support height, including the tray's raised floor.
            support = float(np.quantile(p[:, 2], 0.9) - 0.024)
            support = table_z if support < table_z + 0.003 else support
            pose = estimate_cube_pose(p, table_z=support)
            if pose.rmse > 0.006:
                continue
            detections[color] = Detection(color, pose.position, pose.yaw, pose.rmse, len(p))
        return detections
