#!/usr/bin/env bash
# CPU checks and GPU jobs are sequential. Run from the repository root.
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
python -m pip check
ruff check src tests scripts
pytest -q
workcell calibrate
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WORKCELL_TEST_RENDER=1 pytest -q tests/test_rendering.py
workcell evaluate --base-policy geometric --episodes 20 --seed 20000 --output outputs/basic --video
python scripts/acceptance_basic.py
