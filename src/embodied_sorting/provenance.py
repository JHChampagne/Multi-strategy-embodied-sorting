"""Small local run manifests; no external experiment service required."""

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_manifest():
    from .cli import versions

    root = Path(__file__).resolve().parents[2]
    tree = hashlib.sha256()
    source = root / "src" if (root / "src").is_dir() else Path(__file__).parent
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            tree.update(str(path.relative_to(root)).encode())
            tree.update(path.read_bytes())

    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": versions(),
        "source_sha256": tree.hexdigest(),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command": sys.argv,
        "gpu": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "upstream_lock_sha256": hashlib.sha256((root / "upstream.lock.json").read_bytes()).hexdigest()
        if (root / "upstream.lock.json").is_file()
        else None,
    }
