"""Explicitly privileged Gymnasium training views of the common workstation."""

import gymnasium as gym
import mujoco
import numpy as np

from ..env import WorkstationEnv
from ..control.ik import solve_ik
from ..types import ActionCommand, JAW_OPEN, BLOCK_HALF


class SkillEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, task="reach", max_steps=None, varied_starts=False, render_mode=None):
        if task not in {"reach", "lift"}:
            raise ValueError(task)
        self.station = WorkstationEnv(task, max_steps=100000, render_mode=render_mode)
        self.task = task
        self.max_steps = max_steps or (100 if task == "reach" else 150)
        self.varied_starts = varied_starts
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (24 if task == "reach" else 27,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        s = self.station
        s.reset(seed=seed)
        self.t = 0
        self._held_steps = 0
        if self.task == "reach":
            s._set_block("red", 0.30, 0.12, 0)
            mode = options.get(
                "start_mode", "rest" if self.varied_starts and self.np_random.random() < 0.5 else "home"
            )
            key = s.model.key(mode)
            q = key.qpos[s.qadrs].copy()
            q[:5] += self.np_random.uniform(-0.03, 0.03, 5)
            q = np.clip(q, s.limits[:, 0], s.limits[:, 1])
            q[5] = JAW_OPEN
            s.data.qpos[s.qadrs] = q
            s.data.ctrl[s.act_ids] = q
            mujoco.mj_forward(s.model, s.data)
            if "target" in options:
                self.target = np.asarray(options["target"], dtype=np.float32)
            else:
                for _ in range(100):
                    radius = self.np_random.uniform(0.18, 0.28)
                    angle = self.np_random.uniform(np.deg2rad(-30), np.deg2rad(35))
                    target = np.array(
                        [radius * np.sin(angle), -radius * np.cos(angle), self.np_random.uniform(0.04, 0.15)]
                    )
                    if solve_ik(s.model, s.data, target)[1] < 0.005:
                        self.target = target
                        break
                else:
                    raise RuntimeError("No reachable Reach target")
        else:
            s.data.qpos[s.qadrs[:5]] += self.np_random.uniform(-0.03, 0.03, 5)
            s.data.qpos[s.qadrs[5]] = JAW_OPEN
            s.data.ctrl[s.act_ids] = s.data.qpos[s.qadrs]
        mujoco.mj_forward(s.model, s.data)
        return self._obs(), {"privileged_state": True}

    def _obs(self):
        s = self.station
        q, v, ctrl = s.data.qpos[s.qadrs], s.data.qvel[s.dadrs], s.control
        tcp = s.tcp_pos
        if self.task == "reach":
            return np.r_[q[:5], v[:5], ctrl[:5], tcp, self.target, self.target - tcp].astype(np.float32)
        pos, _ = s.gt_object_pose("red")
        return np.r_[q, v, ctrl, tcp, pos, pos - tcp].astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1).copy()
        if self.task == "reach":
            action[5] = 0
        s = self.station
        s.step(ActionCommand(action, "delta"))
        self.t += 1
        if self.task == "reach":
            distance = float(np.linalg.norm(self.target - s.tcp_pos))
            success = distance < 0.02
            reward = -distance - 0.005 * float(np.square(action).sum()) + float(success)
            terminated = success
        else:
            pos, _ = s.gt_object_pose("red")
            distance = float(np.linalg.norm(pos - s.tcp_pos))
            above = bool(pos[2] > 0.07)
            self._held_steps = self._held_steps + 1 if above else 0
            reward = 0.5 * (1 - np.tanh(10 * distance))
            if distance < 0.03:
                reward += 0.25 * (JAW_OPEN - s.data.qpos[s.qadrs[5]]) / (JAW_OPEN + 0.174)
            reward += 3 * np.clip(pos[2] - BLOCK_HALF, 0, 0.07) / 0.07
            reward += 5 * above - 4 * max(0, pos[2] - 0.14) - 0.005 * np.square(action).sum()
            success = self._held_steps >= 6
            terminated = False
        return (
            self._obs(),
            float(reward),
            terminated,
            self.t >= self.max_steps and not terminated,
            {"is_success": success, "distance": distance, "privileged_state": True},
        )

    def render(self):
        return self.station.render()

    def close(self):
        self.station.close()


class ReachEnv(SkillEnv):
    def __init__(self, **kwargs):
        super().__init__(task="reach", **kwargs)


class LiftEnv(SkillEnv):
    def __init__(self, **kwargs):
        super().__init__(task="lift", **kwargs)
