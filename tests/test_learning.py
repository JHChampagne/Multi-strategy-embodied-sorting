import numpy as np
import pytest

from embodied_sorting.learning.envs import ReachEnv, LiftEnv


@pytest.mark.parametrize("cls,dim", [(ReachEnv, 24), (LiftEnv, 27)])
def test_skill_reset_step_and_markov_control(cls, dim):
    env = cls(max_steps=2)
    try:
        obs, info = env.reset(seed=7)
        assert info["privileged_state"]
        assert obs.shape == (dim,) and env.observation_space.contains(obs)
        ctrl_slice = slice(10, 15) if dim == 24 else slice(12, 18)
        np.testing.assert_allclose(
            obs[ctrl_slice], env.station.control[:5] if dim == 24 else env.station.control
        )
        obs, reward, term, trunc, info = env.step(np.zeros(6))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        _, _, _, trunc, _ = env.step(np.zeros(6))
        assert trunc
    finally:
        env.close()


def test_world_model_rollout_has_no_simulator_access():
    pytest.importorskip("torch")
    from embodied_sorting.learning.dynamics import DynamicsEnsemble, CEMConfig, CEMPlanner
    from embodied_sorting.learning.envs import ReachEnv

    env = ReachEnv()
    try:
        obs, _ = env.reset(seed=5)
        model = DynamicsEnsemble(device="cpu")
        planner = CEMPlanner(
            model,
            CEMConfig(
                horizon=2,
                population=8,
                elites=2,
                iters=1,
                ctrl_lo=env.station.limits[:5, 0],
                ctrl_hi=env.station.limits[:5, 1],
            ),
        )
        env.station.step = lambda *_: (_ for _ in ()).throw(AssertionError("planner touched simulator"))
        a = planner.plan(obs)
        assert a.shape == (6,) and np.isfinite(a).all() and np.max(np.abs(a)) <= 1
        assert a[5] == 0  # the frozen jaw must not enter normalized rollouts as an OOD action
    finally:
        env.close()
