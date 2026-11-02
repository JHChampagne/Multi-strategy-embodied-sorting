"""Original A/C algorithms and scenes, with metrics kept apart from integration."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
for part, package in [("A", "sim2cell"), ("C", "graspsight")]:
    directory = ROOT / ".cache/upstream" / part / "src"
    if not (directory / package).exists():
        raise SystemExit("Run python scripts/fetch_upstream.py first")
    sys.path.insert(0, str(directory))


def main():
    import numpy as np
    from sim2cell.env import PickPlaceEnv
    from sim2cell.expert import ScriptedExpert
    from graspsight.env import GraspEnv
    from graspsight.calibrate import calibrate
    from graspsight.pipeline import GraspSight

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    env = PickPlaceEnv()
    records = []
    try:
        for seed in range(10000, 10020):
            env.rng = np.random.default_rng(seed)
            env.reset()
            records.append({"seed": seed, **ScriptedExpert(env).run()})
    finally:
        env._renderer.close()
    (out / "upstream-A.json").write_text(
        json.dumps(
            {"success_rate": float(np.mean([r["success"] for r in records])), "episodes": records}, indent=2
        )
        + "\n"
    )
    env = GraspEnv()
    records = []
    try:
        calibration = calibrate(env, verbose=False)
        calibration.save(out / "upstream-C-calibration.npz")
        policy = GraspSight(env, out / "upstream-C-calibration.npz")
        for seed in range(10000, 10020):
            env.rng = np.random.default_rng(seed)
            env.reset()
            result = policy.run()
            records.append({"seed": seed, "success": result["success"], "reason": result["reason"]})
    finally:
        env._renderer.close()
    (out / "upstream-C.json").write_text(
        json.dumps(
            {"success_rate": float(np.mean([r["success"] for r in records])), "episodes": records}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
