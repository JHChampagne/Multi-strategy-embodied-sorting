"""The single control-loop owner, with explicit controller handovers and trace records."""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import mujoco

from .control.kinematics import RobotKinematics
from .types import PolicyContext, TaskSpec


class SimulationOracle:
    def __init__(self, env):
        self.env = env

    def object_pose(self, color):
        return self.env.gt_object_pose(color)


class Executor:
    def __init__(self, env, policy, calibration=None):
        self.env = env
        self.policy = policy
        task = TaskSpec(name=env.task, instruction=env.instruction)
        self.context = PolicyContext(
            task,
            RobotKinematics(env.model),
            env.rgbd,
            env.camera_fovy() if env.task == "sorting" else 45.0,
            env.control_hz,
            env.limits.copy(),
            env.model.key("rest").ctrl[env.act_ids].copy(),
            env.trays,
            calibration,
            SimulationOracle(env) if getattr(policy, "mode", None) == "expert" else None,
        )
        self.context.contact_guard = self._unexpected_contacts

    def _unexpected_contacts(self):
        model, data = self.env.model, self.env.data
        moving = {
            model.body(name).id
            for name in (
                "Rotation_Pitch",
                "Upper_Arm",
                "Lower_Arm",
                "Wrist_Pitch_Roll",
                "Fixed_Jaw",
                "Moving_Jaw",
            )
        }
        base = model.body("Base").id
        violations = []
        for i, contact in enumerate(data.contact):
            bodies = [model.geom_bodyid[g] for g in (contact.geom1, contact.geom2)]
            if ((bodies[0] in moving) == (bodies[1] in moving)) or base in bodies:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 2.0:
                violations.append(f"{model.geom(contact.geom1).name}:{model.geom(contact.geom2).name}")
        return violations

    def run(self, seed=10000, trace_path=None, video_path=None, on_step=None):
        obs, info = self.env.reset(seed=seed)
        self.context.seed = seed
        self.policy.reset(self.context)
        writer = None
        trace = None
        started = time.perf_counter()
        latencies = []
        handovers = 0
        controllers = {}
        previous = None
        reason = "time_limit"
        steps = 0
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            trace = open(trace_path, "w")
        if video_path:
            import imageio.v2 as iio

            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            writer = iio.get_writer(str(video_path), fps=self.env.control_hz / 2, codec="libx264")
        try:
            for step in range(self.env.max_steps):
                tick = time.perf_counter()
                command = self.policy.act(obs)
                latencies.append(time.perf_counter() - tick)
                if command is None:
                    reason = getattr(self.policy, "reason", "policy_finished")
                    break
                if command.controller != previous:
                    handovers += previous is not None
                    previous = command.controller
                controllers[command.controller] = controllers.get(command.controller, 0) + 1
                absolute = command.absolute(obs["control"], self.env.limits)
                if on_step:
                    on_step(obs, absolute)
                before = obs
                obs, _, terminated, truncated, info = self.env.step(absolute)
                steps = step + 1
                if trace:
                    trace.write(
                        json.dumps(
                            {
                                "step": step,
                                "sim_time": float(before["time"]),
                                "state": before["state"].tolist(),
                                "control_before": before["control"].tolist(),
                                "raw_action": np.asarray(command.values).tolist(),
                                "kind": command.kind,
                                "applied_action": absolute.tolist(),
                                "controller": command.controller,
                                "phase": command.phase,
                                **command.metadata,
                            }
                        )
                        + "\n"
                    )
                if writer and step % 2 == 0:
                    writer.append_data(self.env.render())
                if terminated:
                    reason = "success"
                    break
                if truncated:
                    break
        finally:
            if writer:
                writer.close()
            if trace:
                trace.close()
        success = bool(info["is_success"])
        return {
            "seed": seed,
            "success": success,
            "placed": info["placed"],
            "steps": steps,
            "wrong_tray": info["wrong_tray"],
            "wrong_sort_count": sum(info["wrong_tray"].values()),
            "sim_seconds": float(self.env.data.time),
            "wall_seconds": time.perf_counter() - started,
            "reason": reason,
            "retries": getattr(self.policy, "retries", 0),
            "handovers": handovers,
            "controllers": controllers,
            "fallback_count": 0,
            "events": getattr(self.policy, "events", []),
            "latency_ms_p50": float(np.median(latencies) * 1000) if latencies else 0,
            "latency_ms_p95": float(np.quantile(latencies, 0.95) * 1000) if latencies else 0,
            "latency_ms_p99": float(np.quantile(latencies, 0.99) * 1000) if latencies else 0,
            "legacy_success": self.env.legacy_success(),
        }
