"""Phase policies yield actions; only Executor is allowed to step the station."""

from __future__ import annotations

import numpy as np
from ..control.ik import HOME_ROLL
from ..perception.pipeline import PerceptionPipeline
from ..types import ActionCommand, JAW_OPEN, JAW_CLOSED, COLORS, BLOCK_HALF, TRAY_HALF


class GeometricPolicy:
    supported_tasks = frozenset({"sorting", "pickplace_compat"})

    def __init__(self, mode="vision", world_model=None, grasp_z=0.015):
        if mode not in {"expert", "vision", "hybrid"}:
            raise ValueError(mode)
        self.grasp_z = grasp_z
        self.mode = mode
        self.supported_tasks = (
            frozenset({"sorting", "pickplace_compat"}) if mode == "expert" else frozenset({"sorting"})
        )
        self.world_model = world_model

    def reset(self, context):
        if context.task.name not in self.supported_tasks:
            raise ValueError("Unsupported task")
        if self.mode == "expert" and context.oracle is None:
            raise ValueError("Expert requires explicitly privileged context")
        if self.mode == "hybrid" and self.world_model is None:
            raise ValueError("Hybrid requires a trained world-model checkpoint")
        self.context = context
        if self.mode == "hybrid":
            import torch

            torch.manual_seed(context.seed)
        self.perception = (
            None if self.mode == "expert" else PerceptionPipeline(context.calibration, context.camera_fovy)
        )
        self.events = []
        self.retries = 0
        self.reason = "running"
        self._generator = self._run()

    def act(self, observation):
        self.obs = observation
        command = next(self._generator, None)
        if command is not None:
            command.metadata.setdefault("material", getattr(self, "current_color", None))
        return command

    def move(self, target, seconds, phase, controller="ik", hold=0):
        start = self.obs["control"].copy()
        n = max(1, round(seconds * self.context.control_hz))
        for i in range(n + round(hold * self.context.control_hz)):
            a = min(1.0, (i + 1) / n)
            yield ActionCommand((1 - a) * start + a * target, controller=controller, phase=phase)

    def waypoint(self, xyz, jaw, roll, seconds, phase, hold=0):
        q = self.context.kinematics.solve(xyz, self.obs["state"], roll)
        yield from self.move(np.r_[q, jaw], seconds, phase, hold=hold)

    def sense(self):
        rest = self.context.rest.copy()
        rest[5] = JAW_OPEN
        yield from self.move(rest, 1.5, "sense_park")
        return self.perception.detect(self.context.rgbd())

    def _in_tray(self, color, position):
        return bool(np.all(np.abs(position[:2] - self.context.trays[color]) < TRAY_HALF - BLOCK_HALF))

    def _pick(self, xy, yaw, tray, color):
        roll, u = self.context.kinematics.grasp_frame(xy, yaw, self.obs["state"])
        x, y = xy
        sx, sy = xy + 0.02 * u
        if self.mode == "hybrid":
            self.world_model.reset()
            target = np.array([sx, sy, 0.10], dtype=np.float32)
            reached = False
            for i in range(100):
                o = self.obs
                tcp = self.context.kinematics.tcp(o["state"])
                distance = float(np.linalg.norm(target - tcp))
                if distance < 0.02:
                    reached = True
                    break
                model_obs = np.r_[
                    o["state"][:5], o["velocity"][:5], o["control"][:5], tcp, target, target - tcp
                ].astype(np.float32)
                a = self.world_model.plan(model_obs)
                a[5] = np.clip((JAW_OPEN - o["control"][5]) / 0.05, -1, 1)
                yield ActionCommand(
                    a,
                    kind="delta",
                    controller="world_model_mpc",
                    phase="pregrasp_reach",
                    metadata={"target": target.tolist(), "distance": distance, "material": color},
                )
                collisions = self.context.contact_guard() if self.context.contact_guard else []
                if collisions:
                    self.events.append(
                        {"material": color, "event": "mpc_unexpected_contact", "contacts": collisions}
                    )
                    return False
            self.events.append(
                {"material": color, "event": "mpc_handover" if reached else "mpc_timeout", "steps": i}
            )
            if not reached:
                return False
        for xyz, jaw, r, duration, phase, hold in [
            ([sx, sy, 0.075], JAW_OPEN, roll, 1.4, "align_hover", 0),
            ([sx, sy, self.grasp_z], JAW_OPEN, roll, 0.9, "descend", 0.1),
            ([x, y, self.grasp_z], JAW_OPEN, roll, 0.6, "slide", 0.1),
            ([x, y, self.grasp_z], JAW_CLOSED, roll, 0, "grasp", 0.6),
            ([x, y, 0.10], JAW_CLOSED, roll, 0.8, "lift", 0.1),
            ([*tray, 0.085], JAW_CLOSED, HOME_ROLL, 1.4, "transport", 0),
            ([*tray, 0.055], JAW_CLOSED, HOME_ROLL, 0.8, "lower", 0),
            ([*tray, 0.055], JAW_OPEN, HOME_ROLL, 0, "release", 0.5),
            ([*tray, 0.085], JAW_OPEN, HOME_ROLL, 0.8, "retreat", 0.5),
        ]:
            yield from self.waypoint(xyz, jaw, r, duration, phase, hold)
        return True

    def _run(self):
        colors = COLORS if self.context.task.name == "sorting" else ("red",)
        for color in colors:
            self.current_color = color
            verified = False
            for attempt in range(3):
                if attempt:
                    self.retries += 1
                if self.mode == "expert":
                    pos, yaw = self.context.oracle.object_pose(color)
                else:
                    try:
                        detections = yield from self.sense()
                    except RuntimeError as error:
                        self.events.append(
                            {
                                "material": color,
                                "attempt": attempt,
                                "event": "perception_error",
                                "error": str(error),
                            }
                        )
                        continue
                    if color not in detections:
                        self.events.append({"material": color, "attempt": attempt, "event": "not_detected"})
                        continue
                    det = detections[color]
                    pos, yaw = det.position, det.yaw
                if self._in_tray(color, pos):
                    verified = True
                    break
                try:
                    executed = yield from self._pick(pos[:2], yaw, self.context.trays[color], color)
                except RuntimeError as error:
                    self.events.append(
                        {"material": color, "attempt": attempt, "event": "ik_error", "error": str(error)}
                    )
                    continue
                if not executed:
                    continue
                if self.mode == "expert":
                    final_pos, _ = self.context.oracle.object_pose(color)
                    verified = self._in_tray(color, final_pos)
                else:
                    try:
                        detections = yield from self.sense()
                    except RuntimeError as error:
                        self.events.append(
                            {
                                "material": color,
                                "attempt": attempt,
                                "event": "verification_error",
                                "error": str(error),
                            }
                        )
                        continue
                    verified = color in detections and self._in_tray(color, detections[color].position)
                self.events.append(
                    {"material": color, "attempt": attempt, "event": "verified" if verified else "retry"}
                )
                if verified:
                    break
            if not verified:
                self.reason = f"failed_{color}"
                return
        # Give the strict evaluator time to observe stable placement and gripper retreat.
        yield from self.move(self.obs["control"], 0.6, "settle")
        self.reason = "completed"
