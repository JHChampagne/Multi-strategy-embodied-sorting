"""Task execution with sensor-only decisions and an independent privileged audit."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np

from .contracts import RuntimeContext, SceneEstimate, PolicyOutput, NoOpOptimizer, ActionCorrection
from .control.kinematics import RobotKinematics
from .evaluation import SortingEvaluator
from .perception.pipeline import PerceptionPipeline
from .policies.base_geometric import select_grasp
from .task import TaskManager
from .types import ActionCommand, JAW_OPEN


class TaskExecutor:
    def __init__(self, env, policy, calibration, optimizer=None, evaluator=True):
        if env.task not in policy.supported_tasks:
            raise ValueError(
                f"Base policy does not support {env.task}; a matching seven-material checkpoint is required"
            )
        self.env, self.policy = env, policy
        self.context = RuntimeContext(
            RobotKinematics(env.model),
            env.rgbd,
            env.model.key("rest").ctrl[env.act_ids].copy(),
            env.limits.copy(),
            env.control_hz,
            env.control_dt,
            env.camera_fovy(),
            calibration,
        )
        self.manager = TaskManager()
        self.context.rest[:] = [-1.6, -3.32, 3.11, 0.5, 0.0, JAW_OPEN]
        self.optimizer = optimizer or NoOpOptimizer()
        self.evaluator = SortingEvaluator(env) if evaluator else None
        self._writer = None
        self._trace = None
        self.policy_latencies = []
        self.perception_latencies = []

    def _step(self, output):
        self.manager.phase = output.phase
        nominal = output.nominal.absolute(self.obs["control"], self.context.limits)
        correction = (
            self.optimizer.optimize(self.obs, nominal.copy(), self.manager.active, output.phase)
            if output.phase is not None and output.phase in getattr(self.optimizer, "enabled_phases", ())
            else ActionCorrection()
        )
        delta = np.asarray(correction.delta, dtype=float)
        if delta.shape != (6,) or not np.isfinite(delta).all():
            raise ValueError("Local correction must be six finite radians")
        if np.any(np.abs(delta) > np.array([0.02] * 5 + [0.05])):
            raise ValueError("Local correction exceeds the nominal-action trust region")
        proposed = np.clip(nominal + delta, self.context.limits[:, 0], self.context.limits[:, 1])
        rate = np.array([0.18] * 5 + [0.5])
        applied = np.clip(proposed, self.obs["control"] - rate, self.obs["control"] + rate)
        before = self.obs
        self.obs, _, _, truncated, _ = self.env.step(applied)
        self.steps += 1
        if self.evaluator is not None:
            self.evaluator.observe()  # return value is intentionally not used by the controller
        goal = self.manager.active
        if self._trace:
            self._trace.write(
                json.dumps(
                    {
                        "episode": self.episode,
                        "record_type": "action",
                        "step": self.steps,
                        "sim_time": float(before["time"]),
                        "goal_id": goal.goal_id if goal else None,
                        "track_id": goal.track_id if goal else None,
                        "slot_id": goal.slot_id if goal else None,
                        "goal": asdict(goal) if goal else None,
                        "phase": output.phase,
                        "controller": output.nominal.controller,
                        "nominal_action": nominal.tolist(),
                        "correction": delta.tolist(),
                        "correction_source": correction.source,
                        "applied_action": applied.tolist(),
                        "state": before["state"].tolist(),
                        "velocity": before["velocity"].tolist(),
                        "control": before["control"].tolist(),
                        "tcp": before["tcp"].tolist(),
                        "next_state": self.obs["state"].tolist(),
                        "next_velocity": self.obs["velocity"].tolist(),
                        "next_tcp": self.obs["tcp"].tolist(),
                    },
                    default=lambda x: x.tolist(),
                )
                + "\n"
            )
        if self._writer and self.steps % 2 == 0:
            self._writer.append_data(self.env.render())
        if truncated:
            raise TimeoutError("episode_timeout")

    def _hold(self, ticks, phase="observe"):
        for _ in range(ticks):
            self._step(
                PolicyOutput(
                    ActionCommand(self.obs["control"], controller="task_manager", phase=phase), phase
                )
            )

    def _park(self):
        target = self.context.rest.copy()
        target[5] = JAW_OPEN
        start = self.obs["control"].copy()
        for i in range(23):
            a = (i + 1) / 23
            self._step(
                PolicyOutput(
                    ActionCommand(
                        (1 - a) * start + a * target, controller="task_manager", phase="sense_park"
                    ),
                    "observe",
                )
            )

    def _sense(self):
        started = time.perf_counter()
        scene = SceneEstimate(self.perception.detect_all(self.context.rgbd()), float(self.obs["time"]))
        self.perception_latencies.append(time.perf_counter() - started)
        if self._trace:
            self._trace.write(
                json.dumps(
                    {
                        "record_type": "scene",
                        "episode": self.episode,
                        "step": self.steps,
                        "scene": asdict(scene),
                    },
                    default=lambda x: x.tolist(),
                )
                + "\n"
            )
        return scene

    def _samples(self):
        scenes = []
        for i in range(3):
            if i:
                self._hold(4)  # 0.264 s: at least 0.2 s at the upstream integration cadence
            scenes.append(self._sense())
        return scenes

    def _execute(self):
        self._park()
        scene = None
        for _ in range(3):
            scene = self._sense()
            if self.manager.initialize(scene):
                break
            self._hold(3)
        else:
            self.manager.fail("initial_instance_count")
            return
        missing_rounds = 0
        while self.manager.state != "failed":
            self.manager.update(scene)
            if all(t.status == "sorted" for t in self.manager.tracks.values()):
                if all(self.manager.final_verify(s) for s in self._samples()):
                    self.manager.state, self.manager.reason = "completed", "completed"
                else:
                    self.manager.fail("final_visual_verification")
                return
            if getattr(self.policy, "requires_grasp_plan", False):
                grasp = select_grasp(self.manager, self.context, self.obs)
            else:
                visible = [t for t in self.manager.tracks.values() if t.status == "pending" and t.visible]
                grasp = (
                    (max(visible, key=lambda t: (t.confidence, t.track_id)), None, None, 0.075, 0.10)
                    if visible
                    else None
                )
            if grasp is None:
                missing_rounds += 1
                if missing_rounds >= 3:
                    self.manager.fail("no_visible_safe_target")
                    return
                self._park()
                scene = self._sense()
                continue
            missing_rounds = 0
            track, roll, standoff, hover, lift = grasp
            goal = self.manager.reserve(track, roll, standoff, hover, lift)
            self.policy.start_goal(goal, self.obs)
            try:
                goal_steps = 0
                previous_phase = None
                while True:
                    started = time.perf_counter()
                    output = self.policy.act(self.obs)
                    self.policy_latencies.append(time.perf_counter() - started)
                    if output is None:
                        break
                    if previous_phase == "lift" and output.phase == "transport":
                        if self.obs["state"][5] < 0.025:
                            raise RuntimeError("empty_grasp_from_jaw_encoder")
                    self._step(output)
                    previous_phase = output.phase
                    goal_steps += 1
                    if output.phase is None and goal_steps % 15 == 0:
                        if self.manager.verify(self._samples(), self.obs["tcp"]):
                            break
                    if goal_steps >= 600:
                        raise RuntimeError("goal_timeout")
            except RuntimeError as error:
                self.manager.retry(f"controller_error:{error}")
            else:
                self.manager.state = "verifying"
                self._park()
                samples = self._samples()
                if self.manager.verify(samples, self.obs["tcp"]):
                    self.manager.complete()
                else:
                    self.manager.retry("placement_not_verified", samples[-1])
                scene = samples[-1]
                continue
            if self.manager.state != "failed":
                self._park()
                scene = self._sense()

    def run(self, seed=0, trace_path=None, video_path=None, on_step=None):
        if on_step is not None:
            raise ValueError(
                "Use trace records for sorting_basic; legacy demonstration callback is unchanged"
            )
        self.episode = seed
        self.obs = None
        self.steps = 0
        self.policy_latencies, self.perception_latencies = [], []
        self.manager.reset()
        self.optimizer.reset()
        self.context.seed = seed
        self.policy.reset_episode(self.context)
        self.perception = PerceptionPipeline(
            self.context.calibration, self.context.camera_fovy, seed, self.env.trays, True
        )
        if self.evaluator is not None:
            self.evaluator.reset()
        started = time.perf_counter()
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            self._trace = open(trace_path, "w")
        if video_path:
            import imageio.v2 as iio

            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self._writer = iio.get_writer(str(video_path), fps=7.5, codec="libx264")
        try:
            self.obs, _ = self.env.reset(seed=seed)
            self._execute()
        except (RuntimeError, ValueError, TimeoutError) as error:
            self.manager.fail(str(error))
        finally:
            if self._trace:
                self._trace.close()
                self._trace = None
            if self._writer:
                self._writer.close()
                self._writer = None
        perceived = self.manager.state == "completed"
        audit = self.evaluator.result() if self.evaluator is not None else {}
        return {
            "seed": seed,
            "perceived_success": perceived,
            "gt_success": audit.get("success"),
            **audit,
            "success": perceived and audit.get("success", perceived),
            "steps": self.steps,
            "reason": self.manager.reason,
            "visual_sorted": sum(t.status == "sorted" for t in self.manager.tracks.values()),
            "retries": sum(max(0, t.attempts - 1) for t in self.manager.tracks.values()),
            "events": self.manager.events,
            "sim_seconds": float(self.obs["time"]) if self.obs is not None else 0.0,
            "wall_seconds": time.perf_counter() - started,
            "policy_latency_ms": self._latencies(self.policy_latencies),
            "perception_latency_ms": self._latencies(self.perception_latencies),
            "model_identity": id(self.env.model),
            "renderer_identities": {str(k): id(v) for k, v in self.env._renderers.items()},
            "failed_attempts": sum(e["event"] == "retry" for e in self.manager.events),
            "fallback_count": 0,
            "tracks": [
                {"track_id": t.track_id, "color": t.color, "status": t.status, "attempts": t.attempts}
                for t in self.manager.tracks.values()
            ],
        }

    @staticmethod
    def _latencies(values):
        return (
            {"mean": float(np.mean(values) * 1000), "p95": float(np.quantile(values, 0.95) * 1000)}
            if values
            else {}
        )

    def run_many(self, episodes=20, seed=20000, output="outputs/basic", video=True):
        if episodes < 1:
            raise ValueError("episodes must be positive")
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        from .provenance import run_manifest

        manifest = run_manifest()
        manifest["calibration"] = (
            {
                "position": self.context.calibration.cam_pos.tolist(),
                "rotation": self.context.calibration.R_wc.tolist(),
                "residual_mm": self.context.calibration.residual_mm,
                "fovy": self.context.camera_fovy,
            }
            if self.context.calibration is not None
            else None
        )
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        records = []
        for i in range(episodes):
            report = self.run(
                seed + i,
                out / "traces" / f"{seed + i}.jsonl",
                out / "episode.mp4" if video and i == 0 else None,
            )
            records.append(report)
            summary = {
                "task": "sorting_basic",
                "base_policy": type(self.policy).__name__,
                "enhancer": type(self.optimizer).__name__,
                "episodes": records,
                "success_rate": sum(r["success"] for r in records) / len(records),
                "item_success_rate": sum(r.get("correct_items", 0) for r in records) / (7 * len(records)),
                "wrong_sort_count": sum(r.get("wrong_sort_count", 0) for r in records),
                "false_positive_count": sum(
                    r["perceived_success"] and not r.get("gt_success", False) for r in records
                ),
                "expected_objects_per_episode": 7,
                "same_process": True,
                "automatic_reset": True,
                "pid": os.getpid(),
                "control_dt": self.context.control_dt,
                "max_steps": self.env.max_steps,
                "task_config": {
                    "trays": self.env.trays,
                    "tray_half": self.env.tray_half,
                    "cube_edge": 0.024,
                    "minimum_spacing": 0.05,
                    "colors": "permuted 3/2/2",
                    "max_extra_retries": 2,
                },
            }
            (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                f"basic seed={seed + i} visual={report['perceived_success']} gt={report.get('gt_success')} "
                f"items={report.get('correct_items', 0)}/7 steps={report['steps']} reason={report['reason']}",
                flush=True,
            )
        return summary
