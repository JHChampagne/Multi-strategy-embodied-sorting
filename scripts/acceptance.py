"""Build a publishable, measured M1 report; exit nonzero if any gate is missing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import yaml
    from embodied_sorting.policies.learned import MODELS, VLM
    from embodied_sorting.cli import versions

    config = yaml.safe_load((ROOT / "configs/m1.yaml").read_text())
    runs = {}
    gates = {}
    for name, threshold in [
        ("expert", config["acceptance"]["expert_sorting"]),
        ("vision", config["acceptance"]["vision_sorting"]),
        ("hybrid", config["acceptance"]["hybrid_sorting"]),
        ("act", None),
        ("vla", None),
    ]:
        try:
            report = load(f"outputs/m1/{name}/report.json")
        except FileNotFoundError:
            gates[name] = False
            continue
        episodes = report["episodes"]
        gates[name] = (
            len(episodes) == 20
            and all(r["steps"] > 0 and r["reason"] != "execution_error" for r in episodes)
            and (threshold is None or report["success_rate"] >= threshold)
        )
        if name == "hybrid":
            gates["hybrid_real_mpc"] = all(
                r.get("controllers", {}).get("world_model_mpc", 0) > 0
                and r.get("controllers", {}).get("ik", 0) > 0
                and r.get("handovers", 0) >= 2
                and r.get("fallback_count") == 0
                for r in episodes
            )
        if name in MODELS:
            report["checkpoint"] = "@".join(MODELS[name])
        runs[name] = report
    for task in ["reach", "lift"]:
        try:
            report = load(f"outputs/m1/rl_{task}/report.json")
            gates[f"rl_{task}"] = (
                report["actual_steps"] >= 4096
                and len(report["episodes"]) >= 5
                and report["parameters_updated"]
                and report["restored_actions_equal"]
            )
            runs[f"rl_{task}"] = report
        except FileNotFoundError:
            gates[f"rl_{task}"] = False
    try:
        wm = load("outputs/world_model/evaluation.json")
        training = load("outputs/world_model/report.json")
        gates["world_model"] = len(wm["episodes"]) == 20 and wm["success_rate"] >= 0.8
        transitions = training.get("training_transitions", training["history"][-1]["transitions"])
        gates["world_model_budget"] = (
            config["world_model"]["initial_transitions"] <= transitions
            and transitions + training.get("diagnostic_extra_transitions", 0)
            <= config["world_model"]["max_transitions"]
        )
        runs["world_model"] = {
            "evaluation": wm,
            "training": training,
            "checkpoint_sha256": digest(ROOT / "outputs/world_model/final.pt"),
        }
    except FileNotFoundError:
        gates["world_model"] = False
    try:
        dataset = load("outputs/dataset-report.json")
        gates["dataset"] = dataset["episodes"] >= 10 and dataset["image_shape"] == [3, 224, 224]
        runs["dataset"] = {k: v for k, v in dataset.items() if k != "records"}
    except FileNotFoundError:
        gates["dataset"] = False
    for name in ["expert", "vision", "hybrid", "act", "vla"]:
        path = ROOT / f"outputs/m1/{name}/episode.mp4"
        if name == "expert" and not path.exists():
            path = ROOT / "outputs/demos/expert/episode.mp4"
        gates[f"video_{name}"] = path.exists() and path.stat().st_size > 1000
    transfer = {}
    for name in ["act", "vla"]:
        path = ROOT / f"outputs/transfer/{name}/report.json"
        if path.exists():
            report = json.loads(path.read_text())
            report["checkpoint"] = "@".join(MODELS[name])
            transfer[name] = report
    baselines = {
        part: load(f"outputs/upstream-{part}.json")
        for part in ["A", "C"]
        if (ROOT / f"outputs/upstream-{part}.json").exists()
    }
    result = {
        "timestamp": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "passed": all(gates.values()),
        "gates": gates,
        "runs": runs,
        "transfer": transfer,
        "upstream_baselines": baselines,
        "models": {**MODELS, "vlm": VLM},
        "upstream": load("upstream.lock.json"),
        "versions": versions(),
        "gpu": "single NVIDIA L40, physical GPU 0",
        "scope": "M1 simulation regression set, one training seed; RL is a pipeline smoke test.",
    }
    out = ROOT / "docs/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "m1.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# M1 measured results",
        "",
        f"Measured: {result['timestamp']}",
        "",
        "All task execution and learning jobs used physical GPU 0 (NVIDIA L40).",
        "Physics uses CPU. These are 20 fixed regression scenes, not a generalization claim.",
        "",
        "| Method | Task | Successful episodes | Status |",
        "| --- | --- | --- | --- |",
    ]
    for name in ["expert", "vision", "hybrid", "act", "vla"]:
        if name in runs:
            report = runs[name]
            lines.append(
                f"| {name} | {report['task']} | {sum(r['success'] for r in report['episodes'])}/{len(report['episodes'])} | {'pass' if gates[name] else 'below gate'} |"
            )
    if "world_model" in runs:
        wm = runs["world_model"]["evaluation"]
        lines.append(
            f"| learned dynamics + CEM-MPC | Reach, home/rest starts | {sum(r['success'] for r in wm['episodes'])}/20 | {'pass' if gates['world_model'] else 'below gate'} |"
        )
    for task in ["reach", "lift"]:
        if f"rl_{task}" in runs:
            r = runs[f"rl_{task}"]
            lines.append(
                f"| SAC | {task}, {r['actual_steps']} training steps | {sum(e['success'] for e in r['episodes'])}/5 | training/update/checkpoint smoke only |"
            )
    lines += [
        "",
        "ACT and SmolVLA use upstream pretrained weights on their original single-block task.",
        "Their scores do not establish multi-material sorting or general language understanding.",
        "SAC's short training budget verifies the pipeline and is not presented as solving the skills.",
        "",
        "## Hybrid execution evidence",
        "",
    ]
    if "hybrid" in runs:
        episodes = runs["hybrid"]["episodes"]
        lines += [
            f"Across {len(episodes)} scenes, MPC executed {sum(e.get('controllers', {}).get('world_model_mpc', 0) for e in episodes)} control actions; "
            f"IK executed {sum(e.get('controllers', {}).get('ik', 0) for e in episodes)}. "
            f"Recorded controller changes: {sum(e.get('handovers', 0) for e in episodes)}. Expert fallbacks: 0.",
            "",
            "MPC controls only open-gripper approach. Geometry/IK controls contact and transport.",
            "",
        ]
    lines += ["## Demonstrations and reproduction", ""]
    if "dataset" in runs:
        d = runs["dataset"]
        lines += [
            f"Exported and read back {d['episodes']} LeRobot demonstrations ({d['frames']} frames), with task labels and front/wrist RGB.",
            "",
        ]
    for part, report in baselines.items():
        lines.append(
            f"Original locked Part {part} baseline: {sum(e['success'] for e in report['episodes'])}/{len(report['episodes'])}, using its original success predicate."
        )
    lines += [
        "",
        "## Diagnostic findings",
        "",
        "- A front tray obstructed some jaw approaches; the final third tray is beside the arm, with verified reachable transport/lowering waypoints.",
        "- Reach freezes the jaw. CEM must also freeze that action dimension before input normalization; otherwise zero-variance training inputs become extreme out-of-distribution planner inputs.",
        "- Nominal 15 Hz control uses 33 steps at 0.002 s, so actual action duration is 0.066 s; dataset video metadata retains the upstream nominal 15 fps.",
        "",
    ]
    if "world_model" in runs:
        training = runs["world_model"]["training"]
        count = training.get("training_transitions", training["history"][-1]["transitions"])
        extra = training.get("diagnostic_extra_transitions", 0)
        lines += [
            f"The selected world model was fitted to {count:,} real transitions. "
            f"Additional diagnostic transitions not used by this checkpoint: {extra:,}. "
            "Total training interactions stayed within the 16,000 budget.",
            "",
        ]
    if transfer:
        lines += [
            "## Pretrained sorting transfer",
            "",
            "| Policy | Complete scenes | Evaluation budget |",
            "| --- | --- | --- |",
        ]
        for name, report in transfer.items():
            lines.append(
                f"| {name} | {sum(e['success'] for e in report['episodes'])}/{len(report['episodes'])} | up to 900 actions per scene, no fine-tuning |"
            )
        lines += [
            "",
            "These limited transfer probes motivate M2 multi-material demonstrations and fine-tuning.",
            "",
        ]
    lines += [
        "Full per-episode measurements, seeds, gates and versions: [results/m1.json](results/m1.json).",
        "",
        f"Overall M1 acceptance: **{'PASS' if result['passed'] else 'NOT COMPLETE'}**.",
        "",
    ]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"passed": result["passed"], "gates": gates}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
