#!/usr/bin/env bash
# Full acceptance run, intentionally sequential on one physical GPU.
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
python -m pip check
ruff check src tests scripts
pytest -q
workcell calibrate
workcell evaluate --policy expert --output outputs/m1/expert --video
workcell evaluate --policy vision --output outputs/m1/vision --video
workcell train-world-model --output outputs/world_model
workcell evaluate-world-model --checkpoint outputs/world_model/final.pt
workcell evaluate --policy hybrid --checkpoint outputs/world_model/final.pt --output outputs/m1/hybrid --video
workcell evaluate --policy act --task pickplace_compat --output outputs/m1/act --video
workcell evaluate --policy vla --task pickplace_compat --output outputs/m1/vla --video
workcell train-rl --task reach --output outputs/m1/rl_reach
workcell train-rl --task lift --output outputs/m1/rl_lift
workcell record
python scripts/acceptance.py
