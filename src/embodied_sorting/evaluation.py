"""Privileged audit, isolated from task decisions and base-policy inputs."""

import numpy as np
import mujoco
from .types import BLOCK_HALF


class SortingEvaluator:
    def __init__(self, station):
        self.station = station
        self.reset()

    def reset(self):
        self.stable = {name: 0 for name in self.station._body_names}
        self.collision_steps = 0

    def observe(self):
        s = self.station
        for name in self.stable:
            self.stable[name] = self.stable[name] + 1 if self._placement(name) else 0
        moving = {
            s.model.body(n).id
            for n in (
                "Rotation_Pitch",
                "Upper_Arm",
                "Lower_Arm",
                "Wrist_Pitch_Roll",
                "Fixed_Jaw",
                "Moving_Jaw",
            )
        }
        for i, c in enumerate(s.data.contact):
            bodies = [s.model.geom_bodyid[g] for g in (c.geom1, c.geom2)]
            if not any(b in moving for b in bodies):
                continue
            # Expected jaw/object contacts are not environmental collisions.
            names = [s.model.body(b).name for b in bodies]
            if any(n.startswith("object_") for n in names):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(s.model, s.data, i, force)
            if force[0] > 2:
                self.collision_steps += 1
                break

    def _placement(self, name):
        s = self.station
        body = s.data.body(name)
        extents = np.abs(body.xmat.reshape(3, 3)) @ np.full(3, BLOCK_HALF)
        color = s._object_colors[name]
        adr = s.model.jnt_dofadr[s.model.joint(name + "_free").id]
        vel = s.data.qvel[adr : adr + 6]
        return bool(
            np.all(np.abs(body.xpos[:2] - s.trays[color]) + extents[:2] < s.tray_half)
            and 0.01 < body.xpos[2] < 0.028
            and np.linalg.norm(vel[:3]) < 0.03
            and np.linalg.norm(vel[3:]) < 0.5
            and (s.tcp_pos[2] > 0.07 or np.linalg.norm(s.tcp_pos[:2] - body.xpos[:2]) > 0.08)
        )

    def result(self):
        placed, wrong, dropped = {}, {}, {}
        s = self.station
        for name, count in self.stable.items():
            p = s.data.body(name).xpos
            placed[name] = count >= 6
            wrong[name] = any(
                c != s._object_colors[name]
                and np.all(np.abs(p[:2] - xy) < s.tray_half - BLOCK_HALF)
                and p[2] < 0.04
                for c, xy in s.trays.items()
            )
            dropped[name] = bool(abs(p[0]) > 0.4 or p[1] < -0.45 or p[1] > 0.15 or p[2] < 0)
        return {
            "success": all(placed.values()),
            "placed": placed,
            "correct_items": sum(placed.values()),
            "wrong_sort_count": sum(wrong.values()),
            "dropped_count": sum(dropped.values()),
            "collision_steps": self.collision_steps,
        }
