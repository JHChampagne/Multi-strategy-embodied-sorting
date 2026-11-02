"""A single-goal RGB-D + IK base controller; task scheduling lives outside it."""

import numpy as np

from ..contracts import PolicyOutput
from ..scene import GRASP_OPEN, STANDOFF_DISTANCE, BASIC_TRAYS
from ..types import ActionCommand, JAW_OPEN, JAW_CLOSED


class GeometricBasePolicy:
    supported_tasks = frozenset({"sorting_basic"})
    requires_grasp_plan = True

    def reset_episode(self, context):
        self.context = context
        self.goal = None
        self._generator = iter(())

    def start_goal(self, goal, observation):
        self.goal, self.obs = goal, observation
        self._generator = self._trajectory()

    def act(self, observation):
        self.obs = observation
        return next(self._generator, None)

    def move(self, target, seconds, phase, hold=0):
        start = self.obs["control"].copy()
        steps = max(1, round(seconds * self.context.control_hz))
        for i in range(steps + round(hold * self.context.control_hz)):
            alpha = min(1.0, (i + 1) / steps)
            action = ActionCommand((1 - alpha) * start + alpha * target, controller="geometric", phase=phase)
            yield PolicyOutput(action, phase)

    def waypoint(self, xyz, jaw, roll, seconds, phase, hold=0):
        q = self.context.kinematics.solve(xyz, self.obs["state"], roll)
        yield from self.move(np.r_[q, jaw], seconds, phase, hold)

    def _trajectory(self):
        g = self.goal
        x, y = g.pick_position[:2]
        sx, sy = g.standoff
        tx, ty = g.place_position
        roll = g.grasp_roll
        bx, by = BASIC_TRAYS[g.color]
        height = 0.075
        place_roll, _ = self.context.kinematics.grasp_frame(
            g.place_position, 0.0, self.obs["state"], height=0.065
        )
        try:
            self.context.kinematics.solve([tx, ty, height], self.obs["state"], place_roll)
        except RuntimeError:
            height = 0.065
        for xyz, jaw, r, sec, phase, hold in [
            ([sx, sy, g.approach_height], GRASP_OPEN, roll, 1.4, "approach", 0.1),
            ([sx, sy, 0.028], GRASP_OPEN, roll, 0.9, "descend", 0.1),
            ([x, y, 0.028], GRASP_OPEN, roll, 0.6, "align", 0.1),
            ([x, y, 0.028], JAW_CLOSED, roll, 0.7, "close", 0.4),
            ([x, y, g.lift_height], JAW_CLOSED, roll, 0.8, "lift", 0.1),
            ([bx, by, 0.085], JAW_CLOSED, place_roll, 1.6, "transport", 0.1),
            ([tx, ty, height], JAW_CLOSED, place_roll, 0.8, "transport", 0.1),
            ([tx, ty, height], JAW_CLOSED, place_roll, 0.8, "place", 0.2),
            ([tx, ty, height], JAW_OPEN, place_roll, 0.5, "release", 0.5),
            ([tx, ty, height], JAW_OPEN, place_roll, 0.9, "retreat", 0.4),
        ]:
            yield from self.waypoint(xyz, jaw, r, sec, phase, hold)


def segment_distance(point, start, end):
    v = end - start
    alpha = np.clip(np.dot(point - start, v) / (np.dot(v, v) + 1e-12), 0, 1)
    return float(np.linalg.norm(point - (start + alpha * v)))


def select_grasp(manager, context, observation):
    """Choose clear perceived objects first, and a collision-free side of their square."""
    visible = [t for t in manager.tracks.values() if t.status == "pending" and t.visible]
    candidates = []
    for track in visible:
        track_candidates = []
        xy = track.position[:2]
        try:
            frames = context.kinematics.grasp_candidates(xy, track.yaw, observation["state"])
        except RuntimeError:
            try:
                frames = context.kinematics.grasp_candidates(
                    xy, track.yaw, observation["state"], height=0.065
                )
            except RuntimeError:
                continue
        for roll, u in frames:
            standoff = xy + STANDOFF_DISTANCE * u
            clearance = min(
                (
                    segment_distance(other.position[:2], standoff, xy)
                    for other in visible
                    if other is not track
                ),
                default=1.0,
            )
            if clearance < 0.028:
                continue
            try:
                hover = next(
                    (z for z in [0.075, 0.065] if reachable(context, [*standoff, z], observation, roll)), None
                )
                lift = next(
                    (
                        z
                        for z in [0.10, 0.085, 0.075, 0.065]
                        if reachable(context, [*xy, z], observation, roll)
                    ),
                    None,
                )
                if hover is None or lift is None:
                    continue
                for xyz in [[*standoff, hover], [*standoff, 0.028], [*xy, 0.028], [*xy, lift]]:
                    q = context.kinematics.solve(xyz, observation["state"], roll)
                    if xyz[2] < 0.05 and context.kinematics.tray_clearance(q, GRASP_OPEN) < 0.005:
                        raise RuntimeError("Fixed tray obstructs open-gripper approach")
                others = [t.position for t in visible if t is not track]
                for alpha in [0, 0.5, 1]:
                    xyz = [*((1 - alpha) * standoff + alpha * xy), 0.028]
                    q = context.kinematics.solve(xyz, observation["state"], roll)
                    if not context.kinematics.clears_visual_objects(q, GRASP_OPEN, others):
                        raise RuntimeError("Open fingers obstruct another visual object")
                # The hinged moving finger sweeps a wider region midway through
                # closing than at either endpoint. Check the complete closure.
                for jaw in [0.3, 0.15, 0.0]:
                    if context.kinematics.tray_clearance(q, jaw) < 0.005:
                        raise RuntimeError("Tray obstructs hinged-finger closure")
                    if not context.kinematics.clears_visual_objects(q, jaw, others):
                        raise RuntimeError("Closing finger obstructs a visual object")
            except RuntimeError:
                continue
            track_candidates.append(
                (clearance, track.confidence, track.track_id, track, roll, standoff, hover, lift)
            )
        if track_candidates:
            track_candidates.sort(key=lambda c: c[0], reverse=True)
            candidates.append(track_candidates[min(track.attempts, len(track_candidates) - 1)])
    if not candidates:
        return None
    chosen = max(candidates, key=lambda c: (c[0], c[1], c[2]))
    return chosen[3:]


def reachable(context, xyz, observation, roll):
    try:
        context.kinematics.solve(xyz, observation["state"], roll)
        return True
    except RuntimeError:
        return False
