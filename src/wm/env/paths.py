from dataclasses import dataclass
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent

_ENV_DIR = _ROOT_DIR / "env"
_TRAIN_DIR = _ROOT_DIR / "train"
_PLAN_DIR = _ROOT_DIR / "plan"

_DATA_DIR = _ENV_DIR / "data"
_LOGS_DIR = _PLAN_DIR / "logs"
MPC_LOG = _LOGS_DIR / "mpc.jsonl"

# episode-major: each frame stored once; the generator writes its metadata next to it as .json
EPISODES_NPZ = _DATA_DIR / "episodes_v2.npz"

# every rollout training run gets its own directory under RUNS_DIR; `latest` is a symlink
# to the most recently started run, so notebooks can always read the ROLLOUT_* paths below
RUNS_DIR = _TRAIN_DIR / "runs"
LATEST_RUN = "latest"


@dataclass(frozen=True)
class RunPaths:
    dir: Path

    @property
    def checkpoint(self) -> Path:
        return self.dir / "rollout.pth"

    @property
    def best_checkpoint(self) -> Path:
        # the epoch with the lowest mean test MSE across rollout steps
        return self.dir / "best.pth"

    @property
    def log(self) -> Path:
        return self.dir / "rollout_log.jsonl"

    @property
    def probe_checkpoint(self) -> Path:
        return self.dir / "probe_checkpoint.pth"

    @property
    def probe_ridge_checkpoint(self) -> Path:
        return self.dir / "probe_ridge_checkpoint.pth"

    @property
    def best_probe_checkpoint(self) -> Path:
        return self.dir / "best_probe_checkpoint.pth"

    @property
    def best_probe_ridge_checkpoint(self) -> Path:
        return self.dir / "best_probe_ridge_checkpoint.pth"

    @property
    def config(self) -> Path:
        return self.dir / "config.json"


def run_paths(name: str = LATEST_RUN) -> RunPaths:
    return RunPaths(RUNS_DIR / name)


_LATEST = run_paths()
ROLLOUT_LOG_PATH = _LATEST.log
ROLLOUT_BEST_CHECKPOINT_PATH = _LATEST.best_checkpoint
ROLLOUT_BEST_PROBE_CHECKPOINT_PATH = _LATEST.best_probe_checkpoint
ROLLOUT_BEST_PROBE_RIDGE_CHECKPOINT_PATH = _LATEST.best_probe_ridge_checkpoint
