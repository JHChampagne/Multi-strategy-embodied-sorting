"""Single command surface; GPU selection precedes MuJoCo and torch imports."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(prog="workcell")
    p.add_argument("--gpu", type=int, default=0)
    sub = p.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--output", default="outputs/calibration.npz")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--policy", choices=["expert", "vision", "hybrid", "act", "vla"], required=True)
    ev.add_argument("--task", choices=["sorting", "pickplace_compat"], default="sorting")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--seed", type=int, default=10000)
    ev.add_argument("--calibration", default="outputs/calibration.npz")
    ev.add_argument("--checkpoint")
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--output", required=True)
    ev.add_argument("--video", action="store_true")
    ev.add_argument("--max-steps", type=int)
    wm = sub.add_parser("train-world-model")
    wm.add_argument("--seed", type=int, default=0)
    wm.add_argument("--transitions", type=int, default=16000)
    wm.add_argument("--epochs", type=int, default=120)
    wm.add_argument("--output", default="outputs/world_model")
    wm.add_argument("--device", default="cuda")
    wm.add_argument("--resume-data", help="Retrain from saved real transitions without collecting them again")
    we = sub.add_parser("evaluate-world-model")
    we.add_argument("--checkpoint", required=True)
    we.add_argument("--episodes", type=int, default=20)
    we.add_argument("--seed", type=int, default=50000)
    we.add_argument("--device", default="cuda")
    we.add_argument("--output", default="outputs/world_model/evaluation.json")
    rl = sub.add_parser("train-rl")
    rl.add_argument("--task", choices=["reach", "lift"], required=True)
    rl.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    rl.add_argument("--steps", type=int, default=4096)
    rl.add_argument("--seed", type=int, default=0)
    rl.add_argument("--device", default="cuda")
    rl.add_argument("--resume")
    rl.add_argument("--output", required=True)
    ds = sub.add_parser("record")
    ds.add_argument("--episodes", type=int, default=10)
    ds.add_argument("--task", choices=["sorting", "pickplace_compat"], default="pickplace_compat")
    ds.add_argument("--root", default="data/m1-demonstrations")
    ds.add_argument("--repo-id", default="local/m1-demonstrations")
    ds.add_argument("--seed", type=int, default=0)
    sub.add_parser("doctor")
    return p


def versions():
    result = {}
    for name in [
        "mujoco",
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "lerobot",
        "transformers",
        "gymnasium",
        "stable-baselines3",
    ]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def evaluate(args):
    import numpy as np
    from .env import WorkstationEnv
    from .execution import Executor
    from .perception.calibrate import Calibration
    from .policies.geometric import GeometricPolicy

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    calibration = Calibration.load(args.calibration) if args.policy in {"vision", "hybrid"} else None
    env = WorkstationEnv(
        args.task,
        images=args.policy in {"act", "vla"},
        max_steps=args.max_steps or (1800 if args.task == "sorting" else 240),
    )
    try:
        if args.policy in {"act", "vla"}:
            from .policies.learned import LeRobotPolicy

            policy = LeRobotPolicy(args.policy, args.checkpoint, args.device)
        else:
            planner = None
            if args.policy == "hybrid":
                from .learning.train import load_world_model

                if not args.checkpoint:
                    raise ValueError("--checkpoint is required for hybrid")
                planner = load_world_model(args.checkpoint, env.limits, args.device)
            policy = GeometricPolicy(args.policy, planner)
        executor = Executor(env, policy, calibration)
        records = []
        for ep in range(args.episodes):
            seed = args.seed + ep
            try:
                report = executor.run(
                    seed,
                    out / "traces" / f"{seed}.jsonl",
                    out / "episode.mp4" if args.video and ep == 0 else None,
                )
            except (RuntimeError, ValueError) as error:
                report = {
                    "seed": seed,
                    "success": False,
                    "placed": env.info()["placed"],
                    "steps": env.info()["steps"],
                    "reason": "execution_error",
                    "error": str(error),
                    "retries": getattr(policy, "retries", 0),
                }
            records.append(report)
            summary = {
                "policy": args.policy,
                "task": args.task,
                "episodes": records,
                "success_rate": float(np.mean([r["success"] for r in records])),
                "single_item_success_rate": float(
                    np.mean([v for r in records for v in r["placed"].values()])
                ),
                "mean_retries": float(np.mean([r["retries"] for r in records])),
                "wrong_sort_rate": float(
                    np.mean([v for r in records for v in r.get("wrong_tray", {}).values()])
                )
                if any("wrong_tray" in r for r in records)
                else None,
                "mean_sim_seconds": float(np.mean([r.get("sim_seconds", 0) for r in records])),
                "versions": versions(),
                "gpu": args.gpu,
                "control_dt": env.control_dt,
                "checkpoint": str(getattr(policy, "checkpoint", args.checkpoint)),
                "capability": "sorting"
                if args.policy in {"expert", "vision", "hybrid"}
                else "pickplace_pretrained; sorting is transfer-only",
            }
            (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                f"{args.policy} seed={seed} success={report['success']} steps={report['steps']} reason={report['reason']}",
                flush=True,
            )
        print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}), flush=True)
        if any(r["reason"] == "execution_error" for r in records):
            raise SystemExit(2)
    finally:
        env.close()


def record(args):
    from .learning.dataset import record_demonstrations

    record_demonstrations(args.root, args.repo_id, args.episodes, args.task, args.seed)


def main():
    args = parser().parse_args()
    needs_gpu = args.command not in {"doctor"} and not (
        args.command == "evaluate" and args.policy == "expert" and not args.video
    )
    if needs_gpu:
        import fcntl

        Path(".cache").mkdir(exist_ok=True)
        gpu_lock = open(f".cache/gpu-{args.gpu}.lock", "w")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "evaluate":
        evaluate(args)
    elif args.command == "calibrate":
        from .env import WorkstationEnv
        from .perception.calibrate import calibrate

        env = WorkstationEnv()
        try:
            calibration = calibrate(env)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            calibration.save(args.output)
            print(json.dumps({"residual_mm": calibration.residual_mm, "path": args.output}))
        finally:
            env.close()
    elif args.command == "train-world-model":
        from .learning.train import train_world_model

        train_world_model(
            args.seed, args.transitions, args.device, args.output, args.epochs, args.resume_data
        )
    elif args.command == "evaluate-world-model":
        import torch
        from .learning.envs import ReachEnv
        from .learning.train import evaluate_skill, load_world_model

        torch.set_num_threads(4)
        torch.manual_seed(0)
        env = ReachEnv(varied_starts=True)
        try:
            planner = load_world_model(args.checkpoint, env.station.limits, args.device)
            report = evaluate_skill(env, planner, args.episodes, args.seed)
            report["checkpoint"] = args.checkpoint
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        finally:
            env.close()
    elif args.command == "train-rl":
        from .learning.train import train_rl

        train_rl(args.task, args.steps, args.seed, args.device, args.output, args.algo, args.resume)
    elif args.command == "record":
        record(args)
    elif args.command == "doctor":
        import subprocess

        print(
            json.dumps(
                {"versions": versions(), "gpu": args.gpu, "render_backend": os.environ["MUJOCO_GL"]}, indent=2
            )
        )
        subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv"], check=False)


if __name__ == "__main__":
    main()
