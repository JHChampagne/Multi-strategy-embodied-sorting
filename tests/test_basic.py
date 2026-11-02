"""Task contracts and runtime tests independent of neural-model downloads."""

from collections import Counter
import numpy as np
import pytest

from embodied_sorting.contracts import SceneEstimate, NoOpOptimizer, PolicyOutput
from embodied_sorting.env import WorkstationEnv
from embodied_sorting.perception.pipeline import Detection
from embodied_sorting.task import TaskManager
from embodied_sorting.types import ActionCommand


def scene():
    return SceneEstimate(
        [
            Detection(c, np.array([-0.09 + 0.035 * i, -0.24, 0.012]), 0.1, 0.0005, 100)
            for i, c in enumerate(["red"] * 3 + ["green"] * 2 + ["yellow"] * 2)
        ],
        0.0,
    )


def test_seven_object_seeded_color_and_reset():
    env = WorkstationEnv("sorting_basic")
    try:
        model_id = id(env.model)
        a, info = env.reset(seed=5)
        colors = env._object_colors.copy()
        poses = env.data.qpos.copy()
        assert sorted(Counter(colors.values()).values()) == [2, 2, 3]
        assert len(env._body_names) == 7
        assert set(info) == {"steps", "control_dt"}
        env.step(a["control"] + 0.01)
        b, _ = env.reset(seed=5)
        np.testing.assert_array_equal(env.data.qpos, poses)
        np.testing.assert_array_equal(a["control"], b["control"])
        assert not env.data.qvel.any() and colors == env._object_colors
        assert id(env.model) == model_id
        env.reset(seed=6)
        assert not np.array_equal(env.data.qpos, poses)
    finally:
        env.close()


def test_same_color_tracking_is_one_to_one():
    manager = TaskManager()
    s = scene()
    assert manager.initialize(s)
    assert len([t for t in manager.tracks.values() if t.color == "red"]) == 3
    ids = set(manager.tracks)
    moved = SceneEstimate(
        [
            Detection(d.color, d.position + [0.001, -0.001, 0], d.yaw, d.rmse, d.points)
            for d in reversed(s.detections)
        ],
        0.2,
    )
    manager.update(moved)
    assert set(manager.tracks) == ids and all(t.visible for t in manager.tracks.values())
    assert len({tuple(t.position) for t in manager.tracks.values()}) == 7
    manager.update(SceneEstimate([], 0.4))
    assert not any(t.visible for t in manager.tracks.values())


def test_slot_reservation_verification_and_bounded_retry():
    manager = TaskManager()
    assert manager.initialize(scene())
    track = next(iter(manager.tracks.values()))
    goal = manager.reserve(track)
    good = Detection(goal.color, np.r_[goal.place_position, 0.018], 0.0, 0.0005, 100)
    samples = [SceneEstimate([good], time) for time in [0.0, 0.264, 0.528]]
    assert manager.verify(samples, np.array([0, 0, 0.1]))
    assert not manager.verify(samples, np.array([0, 0, 0.04]))
    assert not manager.verify([SceneEstimate([], t) for t in [0.0, 0.264, 0.528]], np.array([0, 0, 0.1]))
    manager.complete()
    assert manager.slots[goal.slot_id]["occupied"]
    another = next(t for t in manager.tracks.values() if t.status == "pending" and t.color == track.color)
    reserved = manager.reserve(another)
    assert reserved.slot_id != goal.slot_id
    manager.retry("grasp_missed")
    assert manager.slots[reserved.slot_id]["owner"] is None
    for _ in range(2):
        manager.reserve(another)
        manager.retry("grasp_missed")
    assert manager.state == "failed" and another.attempts == 3
    manager.reset()
    assert not manager.tracks and all(s["owner"] is None for s in manager.slots.values())


def test_unknown_phase_disables_enhancement():
    from embodied_sorting.runtime import TaskExecutor
    from embodied_sorting.policies.base_geometric import GeometricBasePolicy

    class ForbiddenOptimizer:
        def optimize(self, *_):
            raise AssertionError("Enhancement must be disabled without a known phase")

    env = WorkstationEnv("sorting_basic")
    try:
        run = TaskExecutor(
            env, GeometricBasePolicy(), calibration=None, optimizer=ForbiddenOptimizer(), evaluator=False
        )
        run.obs, _ = env.reset(seed=0)
        run.steps, run.episode = 0, 0
        action = run.obs["control"].copy()
        run._step(PolicyOutput(ActionCommand(action), phase=None))
        np.testing.assert_array_equal(run.obs["control"], action)
        assert not hasattr(run.context, "oracle") and not hasattr(run.context, "env")
        np.testing.assert_array_equal(NoOpOptimizer().optimize(None, None, None, None).delta, np.zeros(6))
    finally:
        env.close()


def test_optimizer_phase_mask_and_trust_region():
    from embodied_sorting.runtime import TaskExecutor
    from embodied_sorting.contracts import ActionCorrection
    from embodied_sorting.policies.base_geometric import GeometricBasePolicy

    class ResidualStub:
        enabled_phases = frozenset({"approach"})
        calls = 0
        magnitude = 0.001

        def optimize(self, *_):
            self.calls += 1
            return ActionCorrection(np.array([self.magnitude, 0, 0, 0, 0, 0]), "test")

    env = WorkstationEnv()
    try:
        optimizer = ResidualStub()
        run = TaskExecutor(env, GeometricBasePolicy(), None, optimizer, evaluator=False)
        run.obs, _ = env.reset(seed=0)
        run.steps, run.episode = 0, 0
        original = run.obs["control"].copy()
        run._step(PolicyOutput(ActionCommand(original), "transport"))
        assert optimizer.calls == 0
        run._step(PolicyOutput(ActionCommand(original), "approach"))
        assert optimizer.calls == 1
        assert np.isclose(run.obs["control"][0], original[0] + 0.001)
        optimizer.magnitude = 0.021
        with pytest.raises(ValueError, match="trust region"):
            run._step(PolicyOutput(ActionCommand(original), "approach"))
        assert run.steps == 2  # Invalid corrections never advance simulation.
    finally:
        env.close()


def test_failures_and_timeouts_reset_in_same_process(tmp_path, monkeypatch):
    from embodied_sorting import runtime
    from embodied_sorting.policies.base_geometric import GeometricBasePolicy

    class EmptyPerception:
        def __init__(self, *_):
            pass

        def detect_all(self, *_):
            return []

    monkeypatch.setattr(runtime, "PerceptionPipeline", EmptyPerception)
    env = WorkstationEnv("sorting_basic", max_steps=2)
    try:
        run = runtime.TaskExecutor(env, GeometricBasePolicy(), None, evaluator=False)
        report = run.run_many(3, 0, tmp_path, False)
        assert len(report["episodes"]) == 3
        assert [r["seed"] for r in report["episodes"]] == [0, 1, 2]
        assert all(r["steps"] == 2 and r["reason"] == "episode_timeout" for r in report["episodes"])
        assert all(r["visual_sorted"] == 0 for r in report["episodes"])
    finally:
        env.close()


def test_public_weights_do_not_claim_seven_object_capability():
    from embodied_sorting.policies.registry import make_base_policy

    for name in ["act", "smolvla"]:
        with pytest.raises(ValueError, match="not ready"):
            make_base_policy(name)


def test_uncertain_placement_retains_blocked_slot():
    manager = TaskManager()
    manager.initialize(scene())
    goal = manager.reserve(next(iter(manager.tracks.values())))
    tilted = Detection(goal.color, np.r_[goal.place_position, 0.042], 0.7, 0.001, 100)
    manager.retry("placement_not_verified", SceneEstimate([tilted], 1.0))
    assert manager.slots[goal.slot_id]["owner"] == "blocked"
    assert manager.reason == f"unverified_occupied_slot:{goal.slot_id}"
    assert not manager.slots[goal.slot_id]["occupied"]


def test_unique_completion_capacity_and_final_inventory():
    manager = TaskManager()
    manager.initialize(scene())
    observations = []
    for track in manager.tracks.values():
        goal = manager.reserve(track)
        placed = Detection(goal.color, np.r_[goal.place_position, 0.018], 0.0, 0.0005, 100)
        observations.append(placed)
        snapshots = [SceneEstimate(observations.copy(), t) for t in [1.0, 1.264, 1.528]]
        assert manager.verify(snapshots, np.array([0, 0, 0.1]))
        manager.complete()
    assert len({e["track_id"] for e in manager.events if e["event"] == "sorted"}) == 7
    assert manager.final_verify(SceneEstimate(observations, 2.0))
    assert not manager.final_verify(SceneEstimate(observations[:-1], 2.0))
    duplicate = observations[:-1] + [observations[0]]
    assert not manager.final_verify(SceneEstimate(duplicate, 2.0))
    red = next(t for t in manager.tracks.values() if t.color == "red")
    with pytest.raises(RuntimeError, match="No free slot"):
        manager.reserve(red)


def test_perception_loss_and_initialization_failure_reset(tmp_path, monkeypatch):
    from embodied_sorting import runtime
    from embodied_sorting.policies.base_geometric import GeometricBasePolicy

    class EmptyPerception:
        def __init__(self, *_):
            pass

        def detect_all(self, *_):
            return []

    monkeypatch.setattr(runtime, "PerceptionPipeline", EmptyPerception)
    env = WorkstationEnv("sorting_basic")
    monkeypatch.setattr(env, "rgbd", lambda: {})
    try:
        run = runtime.TaskExecutor(env, GeometricBasePolicy(), None, evaluator=False)
        report = run.run_many(2, 3, tmp_path, False)
        assert all(e["reason"] == "initial_instance_count" for e in report["episodes"])
        assert all(not e["tracks"] for e in report["episodes"])
        assert report["episodes"][0]["model_identity"] == report["episodes"][1]["model_identity"]
    finally:
        env.close()


def test_audit_cannot_drive_visual_completion(monkeypatch):
    from embodied_sorting.runtime import TaskExecutor
    from embodied_sorting.policies.base_geometric import GeometricBasePolicy

    env = WorkstationEnv()
    try:
        run = TaskExecutor(env, GeometricBasePolicy(), None, evaluator=False)
        # Avoid rendering: this test exercises the decision/evaluation boundary.
        monkeypatch.setattr("embodied_sorting.runtime.PerceptionPipeline", lambda *_: None)
        monkeypatch.setattr(run, "_execute", lambda: setattr(run.manager, "state", "completed"))

        class RejectingAudit:
            def reset(self):
                pass

            def result(self):
                return {"success": False, "correct_items": 0}

        run.evaluator = RejectingAudit()
        result = run.run(0)
        assert result["perceived_success"] and not result["gt_success"] and not result["success"]
        assert run.manager.state == "completed"  # No corrective feedback from audit.
    finally:
        env.close()


def test_learned_adapter_resets_at_episode_and_goal(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from embodied_sorting.policies import learned
    from embodied_sorting.policies.registry import LearnedBasePolicy

    class StubAdapter:
        def __init__(self, *_):
            self.queue, self.resets = [], 0

        def reset(self, context):
            self.queue.clear()
            self.resets += 1
            self.context = context

        def act(self, obs):
            return ActionCommand(np.zeros(6))

    monkeypatch.setattr(learned, "LeRobotPolicy", StubAdapter)
    (tmp_path / "task_capabilities.json").write_text(json.dumps({"supported_tasks": ["sorting_basic"]}))
    manager = TaskManager()
    manager.initialize(scene())
    goal = manager.reserve(next(iter(manager.tracks.values())))
    for kind in ["act", "smolvla"]:
        policy = LearnedBasePolicy(kind, tmp_path)
        policy.reset_episode(SimpleNamespace(seed=5))
        policy.adapter.queue.append("stale episode action")
        policy.start_goal(goal, {})
        assert not policy.adapter.queue
        policy.adapter.queue.append("stale goal action")
        policy.start_goal(goal, {})
        assert not policy.adapter.queue and policy.adapter.resets == 3
        assert policy.adapter.context.task.instruction == goal.instruction
        assert policy.act({}).phase is None
