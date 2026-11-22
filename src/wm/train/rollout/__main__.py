import argparse
import json
import sys

import torch

from wm.env.paths import EPISODES_NPZ, LATEST_RUN, run_paths
from wm.nets.jepa import Nets
from wm.train.dataset import N_TEST_EPISODES, RolloutBatches, load_episodes, split_episodes
from wm.train.rollout.rollout_trainer import AUTOCAST_DTYPE, RolloutTrainer
from wm.train.rollout.utils import (
    format_rollout_header,
    format_rollout_row,
    write_training_rollout_to_file,
)
from wm.train.runs import (
    default_run_name,
    resolve_run_name,
    save_checkpoint,
    set_latest,
    start_run,
)

EPOCHS = 200
BATCH_SIZE = 128


def rollout(
    name: str | None = None,
    resume: bool = False,
    seed: int = 0,
    n_train: int | None = None,
    horizon: int = 5,
    epochs: int | None = None,
    patience: int | None = None,
):
    if sys.stdout.isatty():
        print("\033[0m", end="")

    if resume:
        # a resumed run keeps the data settings it started with, so its log stays comparable
        name = resolve_run_name(name or LATEST_RUN)
        paths = run_paths(name)
        config = json.loads(paths.config.read_text(encoding="utf-8"))
        seed, n_train, horizon = config["seed"], config["n_train"], config["horizon"]
        # --epochs / --patience on resume change the run's stopping rule; otherwise keep its own
        overrides = {"epochs": epochs, "patience": patience}
        changed = {k: v for k, v in overrides.items() if v is not None and v != config.get(k)}
        if changed:
            config.update(changed)
            paths.config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        epochs, patience = config["epochs"], config.get("patience")
        set_latest(name)
    else:
        name = name or default_run_name()
        epochs = epochs or EPOCHS
        paths = start_run(
            name,
            {
                "seed": seed,
                "n_train": n_train,
                "n_test": N_TEST_EPISODES,
                "horizon": horizon,
                "epochs": epochs,
                "patience": patience,
                "best_checkpoint_metric": "mean test mse over rollout steps",
                "batch_size": BATCH_SIZE,
                "autocast_dtype": str(AUTOCAST_DTYPE),
                "episodes_npz": str(EPISODES_NPZ),
            },
        )
    print(f"Training rollout run '{name}' in {paths.dir}")
    # we set the global seed for reproducibility
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    start_epoch = 0
    best_test_mse = float("inf")
    best_epoch = -1
    if resume:
        checkpoint = torch.load(paths.checkpoint, weights_only=False, map_location=device)
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_test_mse = checkpoint.get("best_test_mse", best_test_mse)
        # checkpoints from before early stopping have no best_epoch; count from the resume
        best_epoch = checkpoint.get("best_epoch", start_epoch - 1)
        nets = Nets()
        nets.load_nets_checkpoint(checkpoint)
        trainer = RolloutTrainer(device=device, nets=nets)
        trainer.to(device)
        trainer.load_optimizer_checkpoint(checkpoint)
    else:
        nets = Nets()
        trainer = RolloutTrainer(device=device, nets=nets)
        trainer.to(device)

    # frames stay uint8 here; the trainer moves each batch to the device and scales to [0, 1]
    train_episodes, test_episodes = split_episodes(load_episodes(), n_train=n_train)
    print(f"{len(train_episodes)} train episodes, {len(test_episodes)} test episodes")

    # test must be shuffled too: test_epoch computes variances within each batch, and an
    # unshuffled batch holds windows from only ~8 episodes, which shrinks them
    test_loader = RolloutBatches(test_episodes, horizon, BATCH_SIZE, shuffle=True, seed=seed + 1)
    train_loader = RolloutBatches(train_episodes, horizon, BATCH_SIZE, shuffle=True, seed=seed)

    print(format_rollout_header(horizon))
    prev_loss = None
    prev_mse = None
    prev_r2 = None
    for epoch in range(start_epoch, epochs):
        per_step_loss, epoch_variance_loss, epoch_grad_norm = trainer.train_epoch(train_loader)
        mse_losses, r2_scores, copy_z_mse_losses, z_hat_variances, z_target_variances = (
            trainer.test_epoch(test_loader)
        )
        print(
            format_rollout_row(
                epoch,
                per_step_loss,
                epoch_variance_loss,
                epoch_grad_norm,
                mse_losses,
                r2_scores,
                copy_z_mse_losses,
                z_hat_variances,
                z_target_variances,
                prev_loss,
                prev_mse,
                prev_r2,
            )
        )

        write_training_rollout_to_file(
            epoch,
            per_step_loss,
            epoch_variance_loss,
            epoch_grad_norm,
            mse_losses,
            r2_scores,
            copy_z_mse_losses,
            z_hat_variances,
            z_target_variances,
            paths.log,
            resume,
        )

        # one number for both best.pth and early stopping: the mean over rollout steps, which
        # the later (larger) steps dominate
        test_mse = mse_losses.mean().item()
        is_best = test_mse < best_test_mse
        if is_best:
            best_test_mse = test_mse
            best_epoch = epoch

        # saved every epoch (about 2.5 MB), so a resume never re-runs epochs already in the log
        checkpoint = {
            "encoder": nets.encoder.state_dict(),
            "encoder_target": nets.encoder_target.state_dict(),
            "predictor": nets.predictor.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": epoch,
            "test_mse": test_mse,
            "best_test_mse": best_test_mse,
            "best_epoch": best_epoch,
        }
        save_checkpoint(checkpoint, paths.checkpoint)
        if is_best:
            save_checkpoint(checkpoint, paths.best_checkpoint)
            print(f"New best mean test mse {test_mse:.4f}; saved {paths.best_checkpoint}")

        if patience is not None and epoch - best_epoch >= patience:
            print(
                f"Stopping early: no new best mean test mse for {patience} epochs "
                f"(best {best_test_mse:.4f} at epoch {best_epoch}). "
                "To continue, resume with a larger --patience."
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default=None,
        help="Run directory under train/runs (default: a timestamp; with --resume: latest)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="Resume training from checkpoint"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    parser.add_argument(
        "--n-train", type=int, default=None, help="Train episodes to use (default: all)"
    )
    parser.add_argument("--horizon", type=int, default=5, help="Rollout steps per training window")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Epoch budget (default: {EPOCHS}; with --resume: the run's own budget)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Stop after this many epochs without a new best mean test mse (default: off)",
    )
    args = parser.parse_args()

    rollout(
        name=args.name,
        resume=args.resume,
        seed=args.seed,
        n_train=args.n_train,
        horizon=args.horizon,
        epochs=args.epochs,
        patience=args.patience,
    )
