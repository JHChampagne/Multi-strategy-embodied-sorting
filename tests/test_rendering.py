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
