import argparse

from wm.env.paths import LATEST_RUN, run_paths
from wm.probe.runner import train_probe
from wm.train.runs import resolve_run_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the MLP and ridge probes on a frozen JEPA encoder."
    )
    parser.add_argument(
        "--name", default=LATEST_RUN, help="Rollout run under train/runs (default: latest)"
    )
    parser.add_argument(
        "--best",
        action="store_true",
        default=False,
        help="Probe the best-test-mse checkpoint (best.pth) instead of the last one",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    args = parser.parse_args()
    paths = run_paths(resolve_run_name(args.name))
    if args.best:
        checkpoint, probe, ridge = (
            paths.best_checkpoint,
            paths.best_probe_checkpoint,
            paths.best_probe_ridge_checkpoint,
        )
    else:
        checkpoint, probe, ridge = (
            paths.checkpoint,
            paths.probe_checkpoint,
            paths.probe_ridge_checkpoint,
        )
    print(f"Training probes on {checkpoint}")
    train_probe(
        jepa_checkpoint_path=checkpoint,
        probe_checkpoint_path=probe,
        probe_ridge_checkpoint_path=ridge,
        seed=args.seed,
    )
