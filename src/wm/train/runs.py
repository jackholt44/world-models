import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from wm.env.paths import LATEST_RUN, RUNS_DIR, RunPaths, run_paths


def default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_run_name(name: str) -> str:
    """Turn `latest` into the run it points at, so resumes write to the real directory."""
    if name != LATEST_RUN:
        return name
    latest = RUNS_DIR / LATEST_RUN
    if not latest.is_symlink():
        raise FileNotFoundError(f"No runs yet: {latest} does not exist")
    return os.readlink(latest)


def set_latest(name: str) -> None:
    """Point runs/latest at runs/<name>, replacing the old link atomically."""
    tmp = RUNS_DIR / f".{LATEST_RUN}.tmp"
    tmp.unlink(missing_ok=True)
    # relative target, so the link survives moving the repo
    tmp.symlink_to(name, target_is_directory=True)
    os.replace(tmp, RUNS_DIR / LATEST_RUN)


def save_checkpoint(checkpoint: Dict[str, Any], path: Path) -> None:
    """Write to a temp file then rename, so an interrupt mid-write can't corrupt `path`."""
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RUNS_DIR.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def start_run(name: str, config: Dict[str, Any]) -> RunPaths:
    """Create runs/<name>, record its config, and make it the latest run."""
    if name == LATEST_RUN:
        raise ValueError(f"'{LATEST_RUN}' is reserved for the symlink")
    paths = run_paths(name)
    if paths.dir.exists():
        raise FileExistsError(f"Run {name} already exists; pass --resume or pick another --name")
    paths.dir.mkdir(parents=True)
    config = {
        "name": name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        **config,
    }
    paths.config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    set_latest(name)
    return paths
