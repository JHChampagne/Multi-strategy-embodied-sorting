"""Single MuJoCo simulation core; task differences are configuration, not robot forks."""

from __future__ import annotations

from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np

from .types import BLOCK_HALF, COLORS, JOINT_NAMES, TRAY_HALF, TRAYS, ActionCommand
from .control.ik import solve_ik

ASSETS = Path(__file__).parent / "assets" / "so_arm100"


class WorkstationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, task="sorting", images=False, max_steps=None, render_mode=None):
        if task not in {"sorting", "pickplace_compat", "reach", "lift"}:
            raise ValueError(f"Unsupported task: {task}")
        self.task = task
        self.model = mujoco.MjModel.from_xml_path(
            str(ASSETS / ("sorting.xml" if task == "sorting" else "pickplace.xml"))
        )
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(j).id for j in JOINT_NAMES])
        self.qadrs = self.model.jnt_qposadr[self.joint_ids]
        self.dadrs = self.model.jnt_dofadr[self.joint_ids]
        self.act_ids = np.array([self.model.actuator(j).id for j in JOINT_NAMES])
        self.limits = self.model.actuator_ctrlrange[self.act_ids].copy()
        self.control_hz = 15.0
        self.n_substeps = round(1 / (self.control_hz * self.model.opt.timestep))
        self.control_dt = self.n_substeps * self.model.opt.timestep
        self.img_hw = (480, 640)
        self.images = images
        self.render_mode = render_mode
        self.max_steps = max_steps or (1800 if task == "sorting" else 240)
        self.colors = COLORS if task == "sorting" else ("red",)
        self.trays = dict(TRAYS) if task == "sorting" else {"red": TRAYS["red"]}
        self._body_names = {c: f"block_{c}" if task == "sorting" else "block" for c in self.colors}
        self._free_adrs = {
            c: self.model.jnt_qposadr[self.model.joint(n + "_free").id] for c, n in self._body_names.items()
        }
        self._renderers = {}
        self._steps = 0
        self._stable = {c: 0 for c in self.colors}
        self.instruction = (
            "Sort the red, green and yellow blocks into their assigned trays."
            if task == "sorting"
            else "Pick up the red block and place it in the blue tray."
        )
        self.action_space = gym.spaces.Box(
            self.limits[:, 0].astype(np.float32), self.limits[:, 1].astype(np.float32)
        )
        spaces = {
            k: gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
            for k in ("state", "velocity", "control")
        }
        spaces["tcp"] = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        spaces["images"] = gym.spaces.Dict(
            {c: gym.spaces.Box(0, 255, (224, 224, 3), np.uint8) for c in ("front", "wrist")} if images else {}
        )
        spaces["time"] = gym.spaces.Box(0, np.inf, shape=(), dtype=np.float64)
        spaces["task"] = gym.spaces.Text(max_length=256, charset=__import__("string").printable)
        self.observation_space = gym.spaces.Dict(spaces)
        mujoco.mj_forward(self.model, self.data)
        if task == "sorting":
            for color, xy in self.trays.items():
                for z in [0.055, 0.085]:
                    _, error = solve_ik(
                        self.model,
                        self.data,
                        [*xy, z],
                        q_init=self.model.key("home").qpos[self.qadrs[:5]],
                        wrist_roll=-1.57,
                    )
                    if error > 0.003:
                        raise ValueError(f"Tray {color} has unreachable waypoint at z={z}: {error}")

    @property
    def control(self):
        return self.data.ctrl[self.act_ids].copy()

    @property
    def tcp_pos(self):
        return self.data.site("tcp").xpos.copy()

    def _set_block(self, color, x, y, yaw):
        a = self._free_adrs[color]
        self.data.qpos[a : a + 3] = [x, y, BLOCK_HALF]
        self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        home = self.model.key("home")
        self.data.qpos[self.qadrs] = home.qpos[self.qadrs]
        self.data.ctrl[self.act_ids] = home.ctrl[self.act_ids]
        self._steps = 0
        self._stable = {c: 0 for c in self.colors}
        if self.task == "sorting":
            layout = options.get("layout") or self.sample_layout()
            for color in self.colors:
                key = color if color in layout else f"block_{color}"
                self._set_block(color, *layout[key])
            a = self.model.jnt_qposadr[self.model.joint("calib_sphere_free").id]
            self.data.qpos[a : a + 7] = [0.5, 0.3, 0.015, 1, 0, 0, 0]
        else:
            r = self.np_random.uniform(0.19, 0.27)
            th = self.np_random.uniform(np.deg2rad(-20), np.deg2rad(35))
            xy = options.get("block_pos", [r * np.sin(th), -r * np.cos(th)])
            yaw = float(np.arctan2(xy[0], -xy[1]))
            self._set_block("red", *xy, yaw)
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs(), self.info()

    def sample_layout(self):
        placed = []
        layout = {}
        for color in self.colors:
            for _ in range(400):
                r = self.np_random.uniform(0.19, 0.27)
                th = self.np_random.uniform(np.deg2rad(-20), np.deg2rad(35))
                xy = np.array([r * np.sin(th), -r * np.cos(th)])
                if any(np.all(np.abs(xy - np.asarray(t)) < 0.070) for t in self.trays.values()):
                    continue
                if any(np.linalg.norm(xy - p) < 0.060 for p in placed):
                    continue
                yaw = self.np_random.uniform(0, np.pi / 2)
                # Validate the down-facing grasp and hover; all tray waypoints checked separately.
                if any(solve_ik(self.model, self.data, [*xy, z])[1] > 0.003 for z in [0.015, 0.10]):
                    continue
                layout[color] = (*xy, yaw)
                placed.append(xy)
                break
            else:
                raise RuntimeError("Could not sample a separated, reachable sorting layout")
        return layout

    def step(self, action):
        command = action if isinstance(action, ActionCommand) else ActionCommand(action)
        self.data.ctrl[self.act_ids] = command.absolute(self.control, self.limits)
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._steps += 1
        for c in self.colors:
            self._stable[c] = self._stable[c] + 1 if self.placement_valid(c) else 0
        success = self.is_success()
        return (
            self.get_obs(),
            float(success),
            success,
            self._steps >= self.max_steps and not success,
            self.info(),
        )

    def get_obs(self):
        return {
            "state": self.data.qpos[self.qadrs].astype(np.float32).copy(),
            "velocity": self.data.qvel[self.dadrs].astype(np.float32).copy(),
            "control": self.control.astype(np.float32),
            "tcp": self.tcp_pos.astype(np.float32),
            "images": {c: self.render_camera(c, (224, 224)) for c in ("front", "wrist")}
            if self.images
            else {},
            "time": np.float64(self.data.time),
            "task": self.instruction,
        }

    def gt_object_pose(self, color):
        body = self.data.body(self._body_names[color])
        R = body.xmat.reshape(3, 3)
        return body.xpos.copy(), float(np.arctan2(R[1, 0], R[0, 0]))

    def placement_valid(self, color):
        body = self.data.body(self._body_names[color])
        p = body.xpos
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.full(3, BLOCK_HALF)
        center = np.array(self.trays[color])
        adr = self.model.jnt_dofadr[self.model.joint(self._body_names[color] + "_free").id]
        vel = self.data.qvel[adr : adr + 6]
        return bool(
            np.all(np.abs(p[:2] - center) + extent[:2] < TRAY_HALF)
            and 0.01 < p[2] < 0.04
            and np.linalg.norm(vel[:3]) < 0.03
            and np.linalg.norm(vel[3:]) < 0.5
            and self.tcp_pos[2] > 0.07
        )

    def is_success(self):
        return all(v >= 6 for v in self._stable.values())

    def legacy_success(self):
        p, _ = self.gt_object_pose("red")
        return bool(
            np.all(np.abs(p[:2] - self.trays["red"]) < TRAY_HALF) and p[2] < 0.05 and self.tcp_pos[2] > 0.07
        )

    def info(self):
        wrong = {}
        for color in self.colors:
            body = self.data.body(self._body_names[color])
            wrong[color] = any(
                other != color
                and np.all(np.abs(body.xpos[:2] - xy) < TRAY_HALF - BLOCK_HALF)
                and body.xpos[2] < 0.04
                for other, xy in self.trays.items()
            )
        return {
            "is_success": self.is_success(),
            "placed": {c: self._stable[c] >= 6 for c in self.colors},
            "wrong_tray": wrong,
            "steps": self._steps,
            "control_dt": self.control_dt,
        }

    def _renderer(self, hw):
        if hw not in self._renderers:
            self._renderers[hw] = mujoco.Renderer(self.model, height=hw[0], width=hw[1])
        return self._renderers[hw]

    def render_camera(self, camera, hw=(480, 640)):
        r = self._renderer(hw)
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def rgbd(self, camera="rgbd"):
        r = self._renderer(self.img_hw)
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        r.update_scene(self.data, camera=camera)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        try:
            depth = r.render().copy()
        finally:
            r.disable_depth_rendering()
        return {"rgb": rgb, "depth": depth}

    def camera_fovy(self, camera="rgbd"):
        return float(self.model.cam_fovy[self.model.camera(camera).id])

    def render(self):
        return self.render_camera("front")

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
