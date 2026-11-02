#!/usr/bin/env python3
"""Independent seven-object release gate. Keeps failures in the denominator."""

import argparse
import hashlib
import json
from pathlib import Path


def validate(run):
    report = json.loads((run / "report.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    episodes = report["episodes"]
    successes = sum(bool(e["success"]) for e in episodes)
    items = sum(e["correct_items"] for e in episodes)
    trace_checks = []
    hashes = {}
    for ep in episodes:
        path = run / "traces" / f"{ep['seed']}.jsonl"
        actions = []
        valid = True
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["record_type"] != "action":
                    continue
                actions.append(row)
                valid &= row["episode"] == ep["seed"]
                valid &= len(row["nominal_action"]) == len(row["applied_action"]) == 6
                valid &= all(x == 0 for x in row["correction"])
                valid &= row["controller"] in {"geometric", "task_manager"}
                valid &= not any(k in row for k in ["object_pose", "contacts", "gt_success", "segmentation"])
        valid &= len(actions) == ep["steps"]
        completed = [e["track_id"] for e in ep["events"] if e["event"] == "sorted"]
        valid &= len(completed) == len(set(completed))
        if ep["success"]:
            valid &= len(completed) == 7 and ep["perceived_success"] and ep["gt_success"]
        trace_checks.append(bool(valid))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    checks = {
        "exactly_20_unfiltered_episodes": len(episodes) == 20,
        "independent_contiguous_seeds": [e["seed"] for e in episodes] == list(range(20000, 20020)),
        "whole_scene_90_percent": successes >= 18,
        "item_95_percent": items >= 133,
        "success_scenes_have_no_wrong_sort": all(
            e["wrong_sort_count"] == 0 for e in episodes if e["success"]
        ),
        "no_expert_fallback": all(e["fallback_count"] == 0 for e in episodes),
        "single_process_and_model": report["same_process"]
        and len({e["model_identity"] for e in episodes}) == 1,
        "renderer_reused": len({json.dumps(e["renderer_identities"], sort_keys=True) for e in episodes}) == 1,
        "automatic_reset": report["automatic_reset"],
        "independent_geometry_and_noop": report["base_policy"] == "GeometricBasePolicy"
        and report["enhancer"] == "NoOpOptimizer",
        "real_action_traces": all(trace_checks),
        "gpu_0_only": manifest["gpu"] == "0",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scene_successes": successes,
        "scene_total": len(episodes),
        "correct_items": items,
        "item_total": 7 * len(episodes),
        "trace_sha256": hashes,
        "manifest": manifest,
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("outputs/basic"))
    parser.add_argument("--output", type=Path, default=Path("outputs/basic/acceptance.json"))
    args = parser.parse_args()
    result = validate(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in {"passed", "checks", "scene_successes", "correct_items"}
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
