import json
import numpy as np
import mujoco
import pytest

from embodied_sorting.env import WorkstationEnv
from embodied_sorting.types import ActionCommand, JOINT_NAMES
from embodied_sorting.execution import Executor
from embodied_sorting.policies.geometric import GeometricPolicy
from embodied_sorting.control.ik import solve_ik, HOME_ROLL


def test_delta_uses_setpoint_and_clips():
    limits = np.tile([-2.0, 2.0], (6, 1))
    current = np.ones(6) * 1.99
    result = ActionCommand(np.array([1.0, -1.0, 4.0, -4.0, 0.0, 0.0]), "delta").absolute(current, limits)
    np.testing.assert_allclose(result, [2, 1.94, 2, 1.94, 1.99, 1.99])
    for bad in [np.zeros(5), np.full(6, np.nan)]:
        with pytest.raises(ValueError):
            ActionCommand(bad).absolute(current, limits)


def test_seeded_layouts_and_joint_mapping():
    env = WorkstationEnv()
    try:
        a, _ = env.reset(seed=10000)
        poses = {c: env.gt_object_pose(c)[0] for c in env.colors}
        b, _ = env.reset(seed=10000)
        np.testing.assert_array_equal(a["state"], b["state"])
        assert env.observation_space.contains(b)
        for i, name in enumerate(JOINT_NAMES):
            assert env.qadrs[i] == env.model.jnt_qposadr[env.model.joint(name).id]
        for c, p in poses.items():
            np.testing.assert_array_equal(p, env.gt_object_pose(c)[0])
        assert "object_pose" not in b and "segmentation" not in b
        for xy in env.trays.values():
            for z in [0.055, 0.085]:
                assert solve_ik(env.model, env.data, [*xy, z], wrist_roll=HOME_ROLL)[1] < 0.003
    finally:
        env.close()


def test_rotated_footprint_and_stability_are_required():
    env = WorkstationEnv()
    try:
        env.reset(seed=10001)
        xy = np.array(env.trays["red"])
        env._set_block("red", *(xy + [0.023, 0]), np.pi / 4)
        a = env._free_adrs["red"]
        env.data.qpos[a + 2] = 0.018
        # A cube center can be inside while a rotated corner extends beyond the tray.
        mujoco.mj_forward(env.model, env.data)
        assert not env.placement_valid("red")
        env._set_block("red", *xy, 0.0)
        env.data.qpos[a + 2] = 0.018
        mujoco.mj_forward(env.model, env.data)
        assert env.placement_valid("red")
        assert not env.is_success()
    finally:
        env.close()


def test_expert_completes_and_executor_owns_steps(tmp_path):
    env = WorkstationEnv()
    try:
        policy = GeometricPolicy("expert")
        executor = Executor(env, policy)
        assert executor.context.oracle is not None
        result = executor.run(seed=10002, trace_path=tmp_path / "trace.jsonl")
        assert result["success"] and result["fallback_count"] == 0
        trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        assert len(trace) == result["steps"]
        assert {r["controller"] for r in trace} == {"ik"}
        assert np.isclose(trace[1]["sim_time"] - trace[0]["sim_time"], env.control_dt)
    finally:
        env.close()


def test_vision_context_has_no_oracle():
    env = WorkstationEnv()
    try:
        executor = Executor(env, GeometricPolicy("vision"))
        assert executor.context.oracle is None
    finally:
        env.close()
