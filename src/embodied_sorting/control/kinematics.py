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
        self._collision_vertices = {}
        corners = np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T
        for g in range(model.ngeom):
            if model.geom_group[g] != 3:
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                self._collision_vertices[g] = model.mesh_vert[a : a + n].copy()
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                self._collision_vertices[g] = corners * model.geom_size[g]

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

    def grasp_frame(self, xy, yaw, state, height=0.075):
        target = np.array([*xy, height])
        yaw0, _ = self.jaw_frame(target, state, HOME_ROLL)
        yaw1, _ = self.jaw_frame(target, state, HOME_ROLL + 0.2)
        gain = wrap(yaw1 - yaw0, np.pi) / 0.2
        if abs(gain) < 0.1:
            raise RuntimeError("Degenerate jaw orientation Jacobian")
        roll = HOME_ROLL + wrap(yaw - yaw0, np.pi / 2) / gain
        _, u = self.jaw_frame(target, state, roll)
        return roll, u

    def grasp_candidates(self, xy, yaw, state, height=0.075):
        """Equivalent square faces give several physically different slide-in directions."""
        target = np.array([*xy, height])
        yaw0, _ = self.jaw_frame(target, state, HOME_ROLL)
        yaw1, _ = self.jaw_frame(target, state, HOME_ROLL + 0.2)
        gain = wrap(yaw1 - yaw0, np.pi) / 0.2
        if abs(gain) < 0.1:
            return []
        base = HOME_ROLL + wrap(yaw - yaw0, np.pi / 2) / gain
        limits = self.model.jnt_range[self.model.joint("Wrist_Roll").id]
        result = []
        for k in [0, -1, 1, -2, 2]:
            roll = base + k * np.pi / 2 / gain
            if limits[0] <= roll <= limits[1]:
                try:
                    _, u = self.jaw_frame(target, state, roll)
                    result.append((roll, u))
                except RuntimeError:
                    pass
        return result

    def tray_clearance(self, q, jaw):
        """Robot FK against known fixed tray geometry only; no object/contact truth."""
        self.update(np.r_[q, jaw])
        robot = [g for g in range(self.model.ngeom) if self.model.geom_group[g] == 3]
        trays = [g for g in range(self.model.ngeom) if (self.model.geom(g).name or "").startswith("tray_")]
        return min(
            (mujoco.mj_geomDistance(self.model, self.data, a, b, 0.01, None) for a in robot for b in trays),
            default=0.01,
        )

    def clears_visual_objects(self, q, jaw, others):
        """Conservative low-finger swept footprint against perceived cube centers.

        Uses robot collision meshes transformed by FK, and a 20 mm circumscribed
        cube radius including uncertainty. No dynamic object geometry is queried.
        """
        if not others:
            return True
        self.update(np.r_[q, jaw])
        from scipy.spatial import ConvexHull, QhullError

        for g, vertices in self._collision_vertices.items():
            world = vertices @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]
            low = world[world[:, 2] < 0.032, :2]
            if len(low) < 3:
                continue
            try:
                hull = ConvexHull(low)
            except QhullError:
                continue
            for center in others:
                xy = center[:2]
                if np.all(hull.equations[:, :2] @ xy + hull.equations[:, 2] <= 0.020):
                    return False
                # Equation tests overestimate only corners; conservatism is
                # deliberate for the baseline's separated regular cubes.
        return True
