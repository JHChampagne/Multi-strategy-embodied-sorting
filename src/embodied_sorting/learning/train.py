"""Small, resumable single-device learning jobs with explicit held-out evaluations."""

import json
import time
from pathlib import Path
import numpy as np


def evaluate_skill(env, action_fn, episodes=20, seed=50000):
    records = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(action_fn, "reset"):
            action_fn.reset()
        ret = 0.0
        for t in range(env.max_steps):
            a = action_fn.plan(obs) if hasattr(action_fn, "plan") else action_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            if term or trunc:
                break
        records.append(
            {
                "seed": seed + ep,
                "success": bool(info["is_success"]),
                "steps": t + 1,
                "return": float(ret),
                "distance": info["distance"],
            }
        )
    return {"success_rate": float(np.mean([r["success"] for r in records])), "episodes": records}


def train_rl(task, steps=4096, seed=0, device="cuda", output="outputs/rl", algo="sac", resume=None):
    import torch
    from stable_baselines3 import SAC, PPO
    from .envs import SkillEnv

    torch.set_num_threads(4)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    env = SkillEnv(task)
    cls = SAC if algo == "sac" else PPO
    kwargs = dict(seed=seed, device=device, verbose=0, learning_rate=3e-4, gamma=0.98)
    if resume:
        model = cls.load(resume, env=env, device=device)
        replay = Path(str(resume).removesuffix(".zip") + "_replay.pkl")
        if algo == "sac" and replay.exists():
            model.load_replay_buffer(replay)
    elif algo == "sac":
        model = cls("MlpPolicy", env, batch_size=256, buffer_size=100000, learning_starts=1000, **kwargs)
    else:
        model = cls("MlpPolicy", env, n_steps=1024, batch_size=256, **kwargs)
    initial = {n: p.detach().clone() for n, p in model.policy.named_parameters()}
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=resume is None)
        model.save(out / "final")
        if algo == "sac":
            model.save_replay_buffer(out / "final_replay.pkl")
        changed = any(not torch.equal(initial[n], p) for n, p in model.policy.named_parameters())
        restored = cls.load(out / "final", device=device)
        probe, _ = env.reset(seed=50000)
        original_action = model.predict(probe, deterministic=True)[0]
        loaded_action = restored.predict(probe, deterministic=True)[0]
        restored_equal = bool(np.allclose(original_action, loaded_action, atol=1e-6))
        report = evaluate_skill(env, lambda obs: restored.predict(obs, deterministic=True)[0], 5)
        report.update(
            task=task,
            algo=algo,
            seed=seed,
            requested_steps=steps,
            actual_steps=model.num_timesteps,
            parameters_updated=changed,
            restored_actions_equal=restored_equal,
            training_seconds=time.perf_counter() - started,
            device=device,
        )
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if not changed or not restored_equal:
            raise RuntimeError("RL update/checkpoint verification failed")
        return report
    finally:
        env.close()


def load_world_model(checkpoint, limits, device="cuda"):
    import torch
    from .dynamics import DynamicsEnsemble, CEMConfig, CEMPlanner

    model = DynamicsEnsemble(device=device)
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return CEMPlanner(model, CEMConfig(ctrl_lo=limits[:5, 0], ctrl_hi=limits[:5, 1]))


def train_world_model(
    seed=0, max_transitions=16000, device="cuda", output="outputs/world_model", epochs=120, resume_data=None
):
    import torch
    from .dynamics import DynamicsEnsemble, CEMConfig, CEMPlanner, transitions_to_xy
    from .envs import ReachEnv

    if max_transitions < 8000 or max_transitions > 16000:
        raise ValueError("M1 world-model budget must be between 8,000 and 16,000 transitions")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    env = ReachEnv(varied_starts=True)
    env.action_space.seed(seed)
    eval_env = ReachEnv(varied_starts=True)
    model = DynamicsEnsemble(device=device)
    planner = CEMPlanner(
        model, CEMConfig(ctrl_lo=env.station.limits[:5, 0], ctrl_hi=env.station.limits[:5, 1])
    )
    observations, actions, next_observations = [], [], []
    if resume_data:
        with np.load(resume_data) as saved:
            observations = list(saved["obs"])
            actions = list(saved["actions"])
            next_observations = list(saved["next_obs"])
        if not (8000 <= len(actions) <= max_transitions):
            raise ValueError("Replay data must contain 8,000 to max_transitions transitions")
        print(f"Reusing {len(actions)} real transitions from {resume_data}", flush=True)
    history = []
    started = time.perf_counter()
    remaining = max_transitions - len(actions)
    rounds = [0] if resume_data else []
    for count in [3000, 3000, 2000] if resume_data else [8000, 3000, 3000, 2000]:
        count = min(remaining, count)
        if not count:
            break
        rounds.append(count)
        remaining -= count
    try:
        for rnd, count in enumerate(rounds):
            obs, _ = env.reset(seed=int(rng.integers(10000)))
            planner.reset()
            for i in range(count):
                a = (
                    env.action_space.sample()
                    if rnd == 0
                    else np.clip(planner.plan(obs) + rng.normal(0, 0.3, 6), -1, 1).astype(np.float32)
                )
                a[5] = 0
                nobs, _, term, trunc, _ = env.step(a)
                observations.append(obs)
                actions.append(a)
                next_observations.append(nobs)
                obs = nobs
                if term or trunc:
                    obs, _ = env.reset(seed=int(rng.integers(10000)))
                    planner.reset()
                if (i + 1) % 1000 == 0:
                    print(f"round={rnd} collected={i + 1}/{count}", flush=True)
            X, Y = transitions_to_xy(np.array(observations), np.array(actions), np.array(next_observations))
            loss = model.fit(X, Y, epochs=epochs)
            report = evaluate_skill(eval_env, planner, 20, 50000)
            report.update(round=rnd, transitions=len(actions), fit_loss=loss)
            history.append(report)
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "transitions": len(actions),
                    "varied_starts": True,
                    "epochs": epochs,
                },
                out / f"round_{rnd}.pt",
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "transitions": len(actions),
                    "varied_starts": True,
                    "epochs": epochs,
                },
                out / "final.pt",
            )
            (out / "report.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "device": device,
                        "history": history,
                        "seconds": time.perf_counter() - started,
                    },
                    indent=2,
                )
                + "\n"
            )
            np.savez_compressed(
                out / "transitions.npz", obs=observations, actions=actions, next_obs=next_observations
            )
            print(json.dumps(report), flush=True)
            if report["success_rate"] >= 0.8:
                break  # M1 gate met; preserve the interaction budget
        return history
    finally:
        env.close()
        eval_env.close()
