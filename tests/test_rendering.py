"""Run explicitly on an EGL-capable GPU or with MUJOCO_GL=osmesa in CI."""

import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("WORKCELL_TEST_RENDER"), reason="explicit rendering gate")


def test_sensor_snapshot_and_context_cleanup():
    from embodied_sorting.env import WorkstationEnv

    env = WorkstationEnv()
    try:
        env.reset(seed=3)
        t = env.data.time
        rgbd = env.rgbd()
        assert rgbd["rgb"].shape == (480, 640, 3)
        assert rgbd["depth"].shape == (480, 640)
        assert np.isfinite(rgbd["depth"]).all()
        assert env.data.time == t
        rgb = env.render_camera("front", (224, 224))
        assert rgb.dtype == np.uint8 and rgb.std() > 10
    finally:
        env.close()
    assert not env._renderers


def test_seven_object_visual_sort_without_truth_access(tmp_path, monkeypatch):
    from collections import Counter
    from embodied_sorting.env import WorkstationEnv
    from embodied_sorting.runtime import TaskExecutor
    from embodied_sorting.policies.base_geometric import GeometricBasePolicy
    from embodied_sorting.perception.calibrate import calibrate
    from embodied_sorting.evaluation import SortingEvaluator

    env = WorkstationEnv("sorting_basic")
    allowed = {
        "task",
        "model",
        "act_ids",
        "limits",
        "control_hz",
        "control_dt",
        "camera_fovy",
        "rgbd",
        "render",
        "step",
        "reset",
        "max_steps",
        "trays",
        "tray_half",
        "_renderers",
    }

    class SensorsOnly:
        def __getattr__(self, name):
            if name not in allowed:
                raise AssertionError(f"Runtime requested privileged member {name}")
            return getattr(env, name)

    def forbidden(*args, **kwargs):
        raise AssertionError("Privileged truth API used during visual control")

    try:
        calibration = calibrate(env, verbose=False)
        for name in ["gt_object_pose", "placement_valid", "is_success"]:
            monkeypatch.setattr(env, name, forbidden)
        run = TaskExecutor(SensorsOnly(), GeometricBasePolicy(), calibration, evaluator=False)
        report = run.run(0, tmp_path / "trace.jsonl")
        assert report["perceived_success"] and report["gt_success"] is None
        assert Counter(t.color for t in run.manager.tracks.values()).most_common()[0][1] == 3
        assert len({t.track_id for t in run.manager.tracks.values()}) == 7
        assert run.manager.final_verify(run._sense())
        # The independent audit is invoked only AFTER the runtime has finished.
        audit = SortingEvaluator(env)
        assert all(audit._placement(name) for name in audit.stable)
        render_ids = {k: id(v) for k, v in env._renderers.items()}
        env.reset(seed=1)
        env.rgbd()
        assert render_ids == {k: id(v) for k, v in env._renderers.items()}
    finally:
        env.close()
    assert not env._renderers
