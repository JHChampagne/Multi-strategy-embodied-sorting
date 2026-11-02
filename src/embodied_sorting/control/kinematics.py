"""Robot-only FK/IK. Scratch state is populated from encoders, never object poses."""

import mujoco
import numpy as np

from .ik import HOME_ROLL, solve_ik
from ..types import JOINT_NAMES


def wrap(angle, period):
    return (angle + period / 2) % period - period / 2


class RobotKinematics:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.qadrs = [model.jnt_qposadr[model.joint(j).id] for j in JOINT_NAMES]
        self.arm_adrs = self.qadrs[:5]

    def update(self, state):
        self.data.qpos[self.qadrs] = state
        mujoco.mj_kinematics(self.model, self.data)

    def tcp(self, state):
        self.update(state)
        return self.data.site("tcp").xpos.copy()

    def solve(self, target, state, roll=None):
        self.update(state)
        q, error = solve_ik(self.model, self.data, target, q_init=np.asarray(state)[:5], wrist_roll=roll)
        if error > 0.004:
            candidate, other_error = solve_ik(
                self.model,
                self.data,
                target,
                q_init=self.model.key("home").qpos[self.arm_adrs],
                wrist_roll=roll,
            )
            if other_error < error:
                q, error = candidate, other_error
        if error > 0.004:
            raise RuntimeError(f"IK target {np.asarray(target).tolist()} unreachable: {error:.4f} m")
        return q

    def jaw_frame(self, target, state, roll):
        q = self.solve(target, state, roll)
        self.data.qpos[self.arm_adrs] = q
        self.data.qpos[self.qadrs[5]] = 0
        mujoco.mj_kinematics(self.model, self.data)
        fixed = self.data.geom("fixed_jaw_pad_1").xpos
        moving = self.data.geom("moving_jaw_pad_1").xpos
        tcp = self.data.site("tcp").xpos
        axis = fixed[:2] - moving[:2]
        u = fixed[:2] - tcp[:2]
        return float(np.arctan2(axis[1], axis[0])), u / (np.linalg.norm(u) + 1e-12)

    def grasp_frame(self, xy, yaw, state):
        target = np.array([*xy, 0.075])
        yaw0, _ = self.jaw_frame(target, state, HOME_ROLL)
        yaw1, _ = self.jaw_frame(target, state, HOME_ROLL + 0.2)
        gain = wrap(yaw1 - yaw0, np.pi) / 0.2
        if abs(gain) < 0.1:
            raise RuntimeError("Degenerate jaw orientation Jacobian")
        roll = HOME_ROLL + wrap(yaw - yaw0, np.pi / 2) / gain
        _, u = self.jaw_frame(target, state, roll)
        return roll, u
