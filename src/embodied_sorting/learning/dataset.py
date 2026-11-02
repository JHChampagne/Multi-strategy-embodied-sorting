"""Success-filtered LeRobot v3 demonstrations, plus explicit discarded-episode records."""

import json
from pathlib import Path
import numpy as np

from ..types import JOINT_NAMES


def record_demonstrations(root, repo_id, episodes=10, task="pickplace_compat", seed=0):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ..env import WorkstationEnv
    from ..execution import Executor
    from ..policies.geometric import GeometricPolicy

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"Dataset exists: {root}; choose a new output path")
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
    }
    for camera in ["front", "wrist"]:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        features=features,
        root=root,
        robot_type="so_arm100_sim",
        use_videos=True,
        video_backend="pyav",
    )
    env = WorkstationEnv(task, images=True)
    policy = GeometricPolicy("expert")
    records = []
    kept = 0
    try:
        for attempt in range(episodes * 3 + 20):

            def on_step(obs, action):
                instruction = (
                    f"Pick up the {policy.current_color} block and place it in its assigned tray."
                    if task == "sorting"
                    else env.instruction
                )
                frame = {
                    "observation.state": obs["state"],
                    "action": np.asarray(action, dtype=np.float32),
                    "task": instruction,
                }
                frame.update({f"observation.images.{c}": image for c, image in obs["images"].items()})
                dataset.add_frame(frame)

            report = Executor(env, policy).run(seed=seed + attempt, on_step=on_step)
            records.append(report)
            if report["success"]:
                dataset.save_episode()
                kept += 1
                print(f"recorded {kept}/{episodes}", flush=True)
            else:
                dataset.clear_episode_buffer()
            if kept == episodes:
                break
        dataset.finalize()
    finally:
        env.close()
    loaded = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    sample = loaded[0]
    result = {
        "episodes": loaded.num_episodes,
        "frames": loaded.num_frames,
        "sample_keys": sorted(sample.keys()),
        "task": task,
        "records": records,
        "image_shape": list(sample["observation.images.front"].shape),
        "actual_control_dt": env.control_dt,
        "nominal_dataset_fps": 15,
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/dataset-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    if kept != episodes or loaded.num_episodes != episodes:
        raise RuntimeError(f"Only {kept}/{episodes} successful demonstrations")
