"""Visual instance ledger, slot reservations and bounded task-level recovery."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import ObjectTrack, TaskGoal, SceneEstimate
from .scene import tray_slots, BASIC_TRAYS as TRAYS, BASIC_TRAY_HALF as TRAY_HALF
from .types import BLOCK_HALF


class TaskManager:
    def __init__(self, expected_objects=7):
        self.expected_objects = expected_objects
        self.reset()

    def reset(self):
        self.tracks = {}
        self.slots = {
            f"{c}:{i}": {"color": c, "position": xy.copy(), "owner": None, "occupied": False}
            for c, slots in tray_slots().items()
            for i, xy in enumerate(slots)
        }
        self.active = None
        self.events = []
        self.state = "initializing"
        self.reason = None
        self._goal_count = 0
        self.phase = None

    @staticmethod
    def in_any_tray(position):
        return any(np.all(np.abs(position[:2] - xy) < 0.061) for xy in map(np.asarray, TRAYS.values()))

    @staticmethod
    def contained(detection, color):
        extent = BLOCK_HALF * (abs(np.cos(detection.yaw)) + abs(np.sin(detection.yaw)))
        return bool(
            np.all(np.abs(detection.position[:2] - TRAYS[color]) + extent < TRAY_HALF - 0.0005)
            and 0.012 < detection.position[2] < 0.027
        )

    def initialize(self, scene: SceneEstimate):
        detections = [d for d in scene.detections if not self.in_any_tray(d.position)]
        if len(detections) != self.expected_objects:
            return False
        if sorted(sum(d.color == c for d in detections) for c in TRAYS) != [2, 2, 3]:
            return False
        for i, d in enumerate(sorted(detections, key=lambda d: (d.position[0], d.position[1]))):
            name = f"track_{i:03d}"
            self.tracks[name] = ObjectTrack(name, d.color, d.position.copy(), d.yaw, d.confidence)
        self.state = "selecting"
        self.events.append({"event": "initialized", "count": len(self.tracks)})
        return True

    def update(self, scene: SceneEstimate):
        pending = [t for t in self.tracks.values() if t.status != "sorted"]
        detections = [d for d in scene.detections if not self.in_any_tray(d.position)]
        for track in pending:
            track.visible = False
        if not pending or not detections:
            return
        costs = np.array(
            [
                [
                    np.linalg.norm(t.position[:2] - d.position[:2]) if t.color == d.color else 100.0
                    for d in detections
                ]
                for t in pending
            ]
        )
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            if costs[i, j] < 0.075:
                t, d = pending[i], detections[j]
                t.position, t.yaw, t.confidence, t.visible = d.position.copy(), d.yaw, d.confidence, True

    def reserve(self, track, grasp_roll=None, standoff=None, approach_height=0.075, lift_height=0.10):
        free = [(k, v) for k, v in self.slots.items() if v["color"] == track.color and v["owner"] is None]
        if not free:
            raise RuntimeError(f"No free slot for {track.color}")
        key, slot = free[0]
        slot["owner"] = track.track_id
        track.status = "active"
        track.attempts += 1
        self._goal_count += 1
        self.active = TaskGoal(
            f"goal_{self._goal_count:03d}",
            track.track_id,
            track.color,
            track.position.copy(),
            track.yaw,
            key,
            slot["position"].copy(),
            f"Pick the {track.color} block at ({track.position[0]:.3f}, {track.position[1]:.3f}) m "
            f"and place it in {track.color} tray slot {key} at "
            f"({slot['position'][0]:.3f}, {slot['position'][1]:.3f}) m.",
            grasp_roll,
            standoff,
            approach_height,
            lift_height,
        )
        self.state = "executing"
        return self.active

    def verify(self, scenes, tcp):
        goal = self.active
        if goal is None or tcp[2] <= 0.07 or len(scenes) < 3:
            return False
        matches = []
        for scene in scenes[-3:]:
            # Previously completed objects must still occupy their own regions.
            # A displaced old object cannot stand in for a newly placed object.
            if not self._occupied_matches(scene):
                return False
            candidates = [
                d
                for d in scene.detections
                if d.color == goal.color
                and np.linalg.norm(d.position[:2] - goal.place_position) < 0.018
                and self.contained(d, goal.color)
            ]
            if not candidates:
                return False
            matches.append(
                min(candidates, key=lambda d: np.linalg.norm(d.position[:2] - goal.place_position))
            )
        if any(scenes[i + 1].time - scenes[i].time < 0.20 for i in range(len(scenes) - 1)):
            return False
        positions = np.array([d.position for d in matches])
        return bool(np.max(np.linalg.norm(positions - positions.mean(0), axis=1)) < 0.002)

    def complete(self):
        goal = self.active
        self.tracks[goal.track_id].status = "sorted"
        self.tracks[goal.track_id].position = np.r_[goal.place_position, 0.018]
        self.slots[goal.slot_id]["occupied"] = True
        self.events.append({"event": "sorted", "track_id": goal.track_id, "slot_id": goal.slot_id})
        self.active = None
        self.state = "selecting"

    def retry(self, reason, scene=None):
        goal = self.active
        if goal is None:
            return
        track = self.tracks[goal.track_id]
        slot = self.slots[goal.slot_id]
        obstructed = scene is not None and any(
            np.linalg.norm(d.position[:2] - goal.place_position) < 0.032 for d in scene.detections
        )
        slot["owner"] = "blocked" if obstructed else None
        track.status = "pending"
        self.events.append(
            {"event": "retry", "track_id": track.track_id, "attempt": track.attempts, "reason": reason}
        )
        self.active = None
        if obstructed:
            # Preserve occupancy on uncertain deposits. Never send another cube
            # into a physically occupied slot after a failed visual verification.
            self.fail(f"unverified_occupied_slot:{goal.slot_id}")
        elif track.attempts >= 3:
            self.fail(f"retry_exhausted:{track.track_id}")
        else:
            self.state = "observing"

    def fail(self, reason):
        self.state, self.reason = "failed", reason

    def _occupied_matches(self, scene):
        detections = scene.detections
        used = set()
        for slot in self.slots.values():
            if not slot["occupied"]:
                continue
            candidates = [
                (i, d)
                for i, d in enumerate(detections)
                if i not in used
                and d.color == slot["color"]
                and np.linalg.norm(d.position[:2] - slot["position"]) < 0.018
                and self.contained(d, slot["color"])
            ]
            if not candidates:
                return False
            used.add(min(candidates, key=lambda x: np.linalg.norm(x[1].position[:2] - slot["position"]))[0])
        return len(used) == sum(s["occupied"] for s in self.slots.values())

    def final_verify(self, scene):
        if len(self.tracks) != self.expected_objects or any(
            t.status != "sorted" for t in self.tracks.values()
        ):
            return False
        if len(scene.detections) != self.expected_objects or any(
            not self.in_any_tray(d.position) for d in scene.detections
        ):
            return False
        return self._occupied_matches(scene)
