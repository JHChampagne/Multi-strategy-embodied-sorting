"""Calibrated RGB-D geometry and color recognition, without simulator segmentation."""

from dataclasses import dataclass
import numpy as np

from .camera import Intrinsics, backproject, cam_to_world
from .pointcloud import voxel_downsample, ransac_plane, dbscan
from .pose import estimate_cube_pose
from ..types import TRAYS


@dataclass
class Detection:
    color: str
    position: np.ndarray
    yaw: float
    rmse: float
    points: int

    @property
    def confidence(self):
        return float(np.exp(-self.rmse / 0.003) * min(1.0, self.points / 60))


class PerceptionPipeline:
    def __init__(self, calibration, fovy, seed=0, trays=None, strict_trays=False):
        if calibration is None:
            raise ValueError("Vision needs an offline sphere calibration; run workcell calibrate")
        self.trays = trays or TRAYS
        self.strict_trays = strict_trays
        self.calibration = calibration
        self.K = Intrinsics.from_fovy(fovy, width=640, height=480)
        self.rng = np.random.default_rng(seed)

    def detect(self, rgbd):
        """Legacy three-color adapter; new tasks consume detect_all instead."""
        result = {}
        for detection in self.detect_all(rgbd):
            if detection.color not in result or detection.points > result[detection.color].points:
                result[detection.color] = detection
        return result

    def detect_all(self, rgbd):
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
        detections = []
        for color, mask in masks.items():
            p = pts[roi & mask]
            if len(p) < 25:
                continue
            labels = dbscan(p, eps=0.004, min_samples=5)
            clusters = [p[labels == i] for i in range(labels.max() + 1)]
            if not clusters:
                continue
            for p in clusters:
                if len(p) < 25 or np.any(np.ptp(p[:, :2], axis=0) > 0.045):
                    continue
                support = float(np.quantile(p[:, 2], 0.9) - 0.024)
                support = table_z if support < table_z + 0.003 else support
                if any(
                    np.all(np.abs(p[:, :2].mean(0) - center) < (0.060 if self.strict_trays else 0.045))
                    for center in map(np.asarray, self.trays.values())
                ):
                    if self.strict_trays and (
                        np.quantile(p[:, 2], 0.9) > table_z + 0.034 or np.ptp(p[:, 2]) > 0.029
                    ):
                        continue  # Reject stacks/tilted objects rather than forcing them onto the floor.
                    support = table_z + 0.006  # known fixed tray floor, not a simulator pose lookup
                pose = estimate_cube_pose(p, table_z=support)
                if pose.rmse <= 0.006:
                    detections.append(Detection(color, pose.position, pose.yaw, pose.rmse, len(p)))
        return detections
