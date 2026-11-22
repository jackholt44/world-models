from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch
from tqdm import tqdm

from wm.env.paths import LATEST_RUN, run_paths
from wm.nets.inputs import encode_pair, rollout_latent
from wm.nets.jepa import Nets
from wm.train.dataset import (
    AmiBatch,
    AmiDataset,
    EpisodeDataset,
    load_episodes,
    split_episodes,
)
from wm.train.runs import resolve_run_name

NAMES = ("X", "Y", "VX", "VY")
PROBE_EPOCHS = 200  # encoder is frozen; the probe head converges quickly
BATCH_SIZE = 128


def probe_gt_stats(gt_state_t: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-channel mean/std on train GT.
    We use the mean and standard deviation to normalize the ground truth state.

    Args:
        gt_state_t: The ground truth state tensor.

    Returns:
        A tuple of the mean and standard deviation of the ground truth state.
    """

    gt = np.asarray(gt_state_t, dtype=np.float32)
    mean = torch.from_numpy(gt.mean(axis=0))
    std = torch.from_numpy(np.maximum(gt.std(axis=0), 1e-6))
    return mean, std


def normalize_gt(gt: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=gt.device, dtype=gt.dtype)
    std = std.to(device=gt.device, dtype=gt.dtype)
    return (gt - mean) / std


def load_ridge_data(
    checkpoint_path: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ridge beta and bias from the checkpoint file."""
    path = Path(checkpoint_path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"No usable ridge checkpoint at {path}")

    checkpoint = torch.load(path, weights_only=False, map_location=device)
    return checkpoint["beta"], checkpoint["bias"]


def train_probe_epoch(
    probe: torch.nn.Module,
    probe_t_minus_1: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_t_minus_1: torch.optim.Optimizer,
    device: torch.device,
    loader: torch.utils.data.DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    epoch: int,
):
    """Train the probe head on cached encoder features for one epoch.

    ``loader`` yields ``(z, gt_norm, gt_norm_t_minus_1)`` batches — encode once with
    ``collect_data``, then reuse; the frozen encoder is the expensive part.
    """
    probe.train()
    probe_t_minus_1.train()
    loss = torch.tensor(0.0, device=device)
    for z, gt_norm, gt_norm_t_minus_1 in tqdm(
        loader, desc=f"Probe Epoch {epoch} of {PROBE_EPOCHS}"
    ):
        optimizer.zero_grad()
        optimizer_t_minus_1.zero_grad()
        z = z.to(device)
        gt_norm = gt_norm.to(device)
        gt_norm_t_minus_1 = gt_norm_t_minus_1.to(device)
        probe_output = probe(z)
        probe_t_minus_1_output = probe_t_minus_1(z)
        loss = torch.nn.functional.mse_loss(probe_output, gt_norm)
        loss_t_minus_1 = torch.nn.functional.mse_loss(probe_t_minus_1_output, gt_norm_t_minus_1)
        loss.backward()
        optimizer.step()
        loss_t_minus_1.backward()
        optimizer_t_minus_1.step()
    return loss.item()


def collect_data(
    encoder: torch.nn.Module,
    device: torch.device,
    probe_gt_mean: torch.Tensor,
    probe_gt_std: torch.Tensor,
    loader: torch.utils.data.DataLoader[AmiBatch],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect data from the loader.
    Args:
        loader: torch.utils.data.DataLoader[AmiBatch], the loader to collect data from
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], the collected data
    """
    zs: List[torch.Tensor] = []
    gts: List[torch.Tensor] = []
    gts_t_minus_1: List[torch.Tensor] = []

    with torch.no_grad():
        encoder.eval()
        for batch in loader:
            z = encode_pair(encoder, batch.obs_tminus1, batch.obs_t, device)
            gt = normalize_gt(
                batch.gt_state_t.to(device, dtype=torch.float32), probe_gt_mean, probe_gt_std
            )
            gt_t_minus_1 = normalize_gt(
                batch.gt_state_tminus1.to(device, dtype=torch.float32), probe_gt_mean, probe_gt_std
            )
            zs.append(z)
            gts.append(gt)
            gts_t_minus_1.append(gt_t_minus_1)
    return torch.cat(zs, dim=0), torch.cat(gts, dim=0), torch.cat(gts_t_minus_1, dim=0)


def train_probe(
    jepa_checkpoint_path: str,
    probe_checkpoint_path: str,
    probe_ridge_checkpoint_path: str,
    seed: int = 0,
):
    """Train the probe module.
    Load the episodes from the NPZ file and train the probe module.
    For each episode, encode the observations and train the probe module.
    """

    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    nets = Nets()
    nets.to(device)
    checkpoint = torch.load(jepa_checkpoint_path, weights_only=False, map_location=device)

    if not checkpoint.get("encoder"):
        print("Probe train needs a JEPA checkpoint; run python -m wm.train.rollout first.")
        return

    nets.load_nets_checkpoint(checkpoint)

    train_episodes, _ = split_episodes(load_episodes())
    train_dataset = AmiDataset(train_episodes)
    # gt_state_t of every transition is frames 1 .. n_steps - 1 (see Episodes)
    probe_gt_mean, probe_gt_std = probe_gt_stats(
        train_episodes.gt_state[:, 1:-1].reshape(-1, 4).numpy()
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Encoder is frozen: encode once, then train the tiny probe head on (z, gt).
    print("Caching encoder features...")
    zs, gts, gts_t_minus_1 = collect_data(
        nets.encoder, device, probe_gt_mean, probe_gt_std, train_loader
    )
    feature_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(zs.detach(), gts.detach(), gts_t_minus_1.detach()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    probe_optimizer = torch.optim.Adam(nets.probe.parameters(), lr=0.001)
    probe_optimizer_t_minus_1 = torch.optim.Adam(nets.probe_t_minus_1.parameters(), lr=0.001)
    probe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        probe_optimizer, T_max=PROBE_EPOCHS
    )
    probe_scheduler_t_minus_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        probe_optimizer_t_minus_1, T_max=PROBE_EPOCHS
    )

    for epoch in range(PROBE_EPOCHS):
        train_probe_epoch(
            nets.probe,
            nets.probe_t_minus_1,
            probe_optimizer,
            probe_optimizer_t_minus_1,
            device,
            feature_loader,
            epoch,
        )
        probe_scheduler.step()
        probe_scheduler_t_minus_1.step()

    probe_checkpoint = {
        "probe": nets.probe.state_dict(),
        "probe_t_minus_1": nets.probe_t_minus_1.state_dict(),
        "probe_optimizer": probe_optimizer.state_dict(),
        "probe_optimizer_t_minus_1": probe_optimizer_t_minus_1.state_dict(),
        "probe_gt_mean": probe_gt_mean,
        "probe_gt_std": probe_gt_std,
    }
    torch.save(probe_checkpoint, probe_checkpoint_path)

    alpha = 1e-3
    beta, bias = fit_ridge_regression(zs, gts, alpha=alpha)

    ridge_checkpoint = {
        "beta": beta,
        "bias": bias,
        "alpha": alpha,
    }

    torch.save(ridge_checkpoint, probe_ridge_checkpoint_path)
    print(f"Saved MLP probe to {probe_checkpoint_path}")
    print(f"Saved ridge probe to {probe_ridge_checkpoint_path}")


def _load_nets_and_probe(
    nets: Nets, jepa_checkpoint_path: str, probe_checkpoint_path: str, device: torch.device
) -> tuple[Nets, torch.Tensor, torch.Tensor]:
    """Load JEPA and MLP probe weights into `nets`, in eval mode, plus the probe's GT stats."""
    checkpoint = torch.load(jepa_checkpoint_path, weights_only=False, map_location=device)

    if not checkpoint.get("encoder"):
        raise RuntimeError(
            "Probe eval needs a JEPA checkpoint; run python -m wm.train.rollout first."
        )

    probe_checkpoint = torch.load(probe_checkpoint_path, weights_only=False, map_location=device)
    probe_gt_mean = probe_checkpoint["probe_gt_mean"]
    probe_gt_std = probe_checkpoint["probe_gt_std"]

    nets.to(device)

    nets.load_nets_checkpoint(checkpoint)
    nets.probe.load_state_dict(probe_checkpoint["probe"], strict=True)
    nets.probe_t_minus_1.load_state_dict(probe_checkpoint["probe_t_minus_1"], strict=True)
    nets.eval()
    return nets, probe_gt_mean, probe_gt_std


def load_eval(
    nets: Nets,
    jepa_checkpoint_path: str,
    probe_checkpoint_path: str,
    probe_ridge_checkpoint_path: str,
    device: torch.device | None = None,
) -> tuple[Nets, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load frozen JEPA + probe + ridge weights for notebook eval."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    nets, probe_gt_mean, probe_gt_std = _load_nets_and_probe(
        nets, jepa_checkpoint_path, probe_checkpoint_path, device
    )
    beta, bias = load_ridge_data(probe_ridge_checkpoint_path, device)
    return nets, beta, bias, probe_gt_mean, probe_gt_std


class LoadedRun(NamedTuple):
    """A trained run ready to plan with: the nets plus the probe's GT normalization."""

    nets: Nets
    probe_gt_mean: torch.Tensor  # (4,)
    probe_gt_std: torch.Tensor  # (4,)
    device: torch.device

    def decode_state(self, trajectories: torch.Tensor) -> torch.Tensor:
        """MLP probe in world units: (..., latent) -> (..., 4) as (x, y, vx, vy).

        Positions are in pixels and velocities in pixels per step, like gt_state. Use the
        MLP probe rather than ridge: the latent stores state on a curved surface, and a
        linear readout misses part of it (about 3.5 px position RMSE against 0.7 px).
        """
        mean = self.probe_gt_mean.to(trajectories)
        std = self.probe_gt_std.to(trajectories)
        return self.nets.probe(trajectories) * std + mean


def load_run(
    name: str = LATEST_RUN, best: bool = True, device: torch.device | None = None
) -> LoadedRun:
    """Load a rollout run and its MLP probe by run name; `latest` follows the symlink.

    best=True loads best.pth with the probe trained on it (python -m wm.probe --best);
    best=False loads the last epoch's rollout.pth with its probe (python -m wm.probe).
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    paths = run_paths(resolve_run_name(name))
    if best:
        checkpoint_path, probe_path = paths.best_checkpoint, paths.best_probe_checkpoint
    else:
        checkpoint_path, probe_path = paths.checkpoint, paths.probe_checkpoint
    if not probe_path.exists():
        flag = " --best" if best else ""
        raise FileNotFoundError(
            f"No probe at {probe_path}; train one with python -m wm.probe --name {name}{flag}"
        )

    nets, probe_gt_mean, probe_gt_std = _load_nets_and_probe(
        Nets(), checkpoint_path, probe_path, device
    )
    return LoadedRun(nets, probe_gt_mean, probe_gt_std, device)


def test_loader() -> torch.utils.data.DataLoader:
    _, test_episodes = split_episodes(load_episodes())
    return torch.utils.data.DataLoader(
        AmiDataset(test_episodes), batch_size=BATCH_SIZE, shuffle=False
    )


def is_wall_bounce(gt_state_t: torch.Tensor, gt_state_tp1: torch.Tensor) -> torch.Tensor:
    """True when position did not follow x += vx * dt (dt=1, v-then-x integrator)."""
    res_x = gt_state_tp1[:, 0] - gt_state_t[:, 0] - gt_state_tp1[:, 2]
    res_y = gt_state_tp1[:, 1] - gt_state_t[:, 1] - gt_state_tp1[:, 3]
    return (res_x.abs() > 1e-2) | (res_y.abs() > 1e-2)


def bounce_splits(bounce: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones(len(bounce), dtype=bool),
        "bounce": bounce,
        "free": ~bounce,
    }


def calculate_probe_mse_per_step(z_decoded: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Calculate the MSE of the z_decoded compared to the gt."""
    # (B, T, 4) -> (T, 4)
    return ((z_decoded - gt) ** 2).mean(axis=(0))


def calculate_latent_mse_per_step(trajectory: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    # per step mse

    # çollapse batch then z, leaving T
    mse = ((trajectory - z_target) ** 2).mean(axis=(0, 2))
    return mse


def calculate_per_dim_variance(trajectory: np.ndarray) -> np.ndarray:
    return np.var(trajectory, axis=(0))  # (B,T,32) => (T, 32)


def process_batch(
    nets: Nets,
    device: torch.device,
    batch: AmiBatch,
    probe_gt_mean: torch.Tensor,
    probe_gt_std: torch.Tensor,
    shuffle_actions: bool = False,
):
    # batch fields are (B, T, ...): every transition of each episode
    z0 = encode_pair(nets.encoder, batch.obs_tminus1[:, 0], batch.obs_t[:, 0], device)
    z0_decoded = nets.probe(z0)
    actions = batch.action.to(device, dtype=torch.float32)  # (B, T, 2)
    if shuffle_actions:
        actions = actions.roll(1, dims=0)

    trajectory = rollout_latent(nets.predictor, z0, actions)  # (B, T, z), column h = h+1 ahead
    z_decoded = nets.probe(trajectory)  # (B, T, 4)
    z_target = encode_pair(nets.encoder_target, batch.obs_t, batch.obs_tp1, device)  # (B, T, z)
    gt_t = normalize_gt(
        batch.gt_state_tp1.to(device, dtype=torch.float32), probe_gt_mean, probe_gt_std
    )  # (B, T, 4)
    hold = z0.unsqueeze(1).expand_as(z_target)  # copy-z0 baseline at every h
    hold_decoded = z0_decoded.unsqueeze(1).expand_as(gt_t)  # (B, T, 4)
    return trajectory, z_target, hold, hold_decoded, z_decoded, gt_t


def rollout(
    nets: Nets,
    device: torch.device,
    probe_gt_mean: torch.Tensor,
    probe_gt_std: torch.Tensor,
    shuffle_actions: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    with torch.no_grad():
        _, test_episodes = split_episodes(load_episodes())
        dataset = EpisodeDataset(test_episodes)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

        zs, zts, holds, hold_decodeds, z_decodeds, gts = [], [], [], [], [], []

        for batch in tqdm(loader, desc="Rollout"):
            z, zt, hold, hold_decoded, z_decoded, gt = process_batch(
                nets,
                device,
                batch,
                probe_gt_mean,
                probe_gt_std,
                shuffle_actions=shuffle_actions,
            )

            zs.append(z)
            zts.append(zt)
            holds.append(hold)
            hold_decodeds.append(hold_decoded)
            z_decodeds.append(z_decoded)
            gts.append(gt)

        return (
            torch.cat(zs, dim=0).detach().cpu().numpy(),
            torch.cat(zts, dim=0).detach().cpu().numpy(),
            torch.cat(holds, dim=0).detach().cpu().numpy(),
            torch.cat(hold_decodeds, dim=0).detach().cpu().numpy(),
            torch.cat(z_decodeds, dim=0).detach().cpu().numpy(),
            torch.cat(gts, dim=0).detach().cpu().numpy(),
        )


@torch.no_grad()
def collect_eval(
    nets: Nets,
    beta: torch.Tensor,
    bias: torch.Tensor,
    probe_gt_mean: torch.Tensor,
    probe_gt_std: torch.Tensor,
    loader: torch.utils.data.DataLoader[AmiBatch],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """One test pass: encoder z, predictor z_hat, EMA z_target, probes, GT, bounce."""
    buckets: Dict[str, list] = {
        "z": [],
        "z_hat": [],
        "z_target": [],
        "z_decoded": [],
        "z_decoded_t_minus_1": [],
        "z_hat_decoded": [],
        "z_target_decoded": [],
        "ridge_decoded": [],
        "gt_t": [],
        "gt_tminus1": [],
        "gt_tp1": [],
        "bounce": [],
        "action": [],
        "t": [],
    }

    for batch in tqdm(loader, desc="Eval probe"):
        action = batch.action.to(device, dtype=torch.float32)
        gt_t = batch.gt_state_t.to(device, dtype=torch.float32)
        gt_tminus1 = batch.gt_state_tminus1.to(device, dtype=torch.float32)
        gt_tp1 = batch.gt_state_tp1.to(device, dtype=torch.float32)

        z = encode_pair(nets.encoder, batch.obs_tminus1, batch.obs_t, device)
        z_target = encode_pair(nets.encoder_target, batch.obs_t, batch.obs_tp1, device)
        z_hat = nets.predictor(torch.cat([z, action], dim=1))

        buckets["z"].append(z.cpu().numpy())
        buckets["z_hat"].append(z_hat.cpu().numpy())
        buckets["z_target"].append(z_target.cpu().numpy())
        buckets["z_decoded"].append(nets.probe(z).cpu().numpy())
        buckets["z_hat_decoded"].append(nets.probe(z_hat).cpu().numpy())
        buckets["z_target_decoded"].append(nets.probe(z_target).cpu().numpy())
        buckets["z_decoded_t_minus_1"].append(nets.probe_t_minus_1(z).cpu().numpy())
        buckets["ridge_decoded"].append((z @ beta.to(z) + bias.to(z)).cpu().numpy())
        buckets["gt_t"].append(normalize_gt(gt_t, probe_gt_mean, probe_gt_std).cpu().numpy())
        buckets["gt_tminus1"].append(
            normalize_gt(gt_tminus1, probe_gt_mean, probe_gt_std).cpu().numpy()
        )
        buckets["gt_tp1"].append(normalize_gt(gt_tp1, probe_gt_mean, probe_gt_std).cpu().numpy())
        buckets["bounce"].append(is_wall_bounce(batch.gt_state_t, batch.gt_state_tp1).numpy())
        buckets["action"].append(action.cpu().numpy())
        buckets["t"].append(batch.t.cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in buckets.items()}


def r2_mse(
    pred: np.ndarray, gt: np.ndarray, probe_gt_std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column R² and MSE. ``pred`` / ``gt`` are (N, K)."""
    r2 = np.empty(pred.shape[1], dtype=np.float64)
    mse = np.empty(pred.shape[1], dtype=np.float64)
    pixel_rmse = np.empty(pred.shape[1], dtype=np.float64)
    for i in range(pred.shape[1]):
        p, g = pred[:, i], gt[:, i]
        ss_var = np.sum((g - g.mean()) ** 2)
        ss_res = np.sum((p - g) ** 2)
        r2[i] = 1 - ss_res / ss_var
        mse[i] = np.mean((p - g) ** 2)
        # because we used z-scores we need to multiply the
        # root mse by the sd of the measured quantity to get the pixel rmse
        pixel_rmse[i] = np.sqrt(mse[i]) * probe_gt_std[i]
    return r2, mse, pixel_rmse


def print_repr_table(
    preds: Dict[str, np.ndarray],
    gt: np.ndarray,
    probe_gt_std: np.ndarray,
    names: Tuple[str, ...] = NAMES,
) -> None:
    header = f"{'dim':<5}{'probe':<12}{'R2':>12}{'MSE':>12}{'Pixel RMSE':>12}"
    print(header)
    print("-" * len(header))
    scores = {label: r2_mse(pred, gt, probe_gt_std) for label, pred in preds.items()}
    for i, name in enumerate(names):
        for j, (label, (r2, mse, pixel_rmse)) in enumerate(scores.items()):
            print(
                f"{name if j == 0 else '':<5}{label:<12}{r2[i]:12.4f}{mse[i]:12.4f}{pixel_rmse[i]:12.4f}"
            )
        print()


def print_latent_mse(
    splits: Dict[str, np.ndarray], z_target: np.ndarray, z_hat: np.ndarray, z: np.ndarray
) -> None:
    print("latent MSE")
    header = (
        f"{'split':8} {'n':>6} {'z_hat MSE':>8} {'z MSE':>8} "
        f"{'z_tgt var':>8} {'r2_hat':>8} {'r2_z':>8} {'r2_mean':>8}"
    )
    print(header)
    print("-" * len(header))
    for split, mask in splits.items():
        z_hat_mse = np.mean((z_hat[mask] - z_target[mask]) ** 2)
        z_mse = np.mean((z[mask] - z_target[mask]) ** 2)
        z_tgt_var = np.mean(np.var(z_target[mask], axis=0))
        r2_hat = 1 - z_hat_mse / z_tgt_var
        r2_z = 1 - z_mse / z_tgt_var
        # the mse of the mean is the variance of the target
        z_mean_mse = z_tgt_var
        r2_mean = 1 - z_mean_mse / z_tgt_var
        print(
            f"{split:8} {int(mask.sum()):6d}  {z_hat_mse:>8.4f} {z_mse:>8.4f} "
            f"{z_tgt_var:>8.4f} {r2_hat:>8.4f} {r2_z:>8.4f} {r2_mean:>8.4f}"
        )


def print_dynamics_table(
    splits: Dict[str, np.ndarray],
    pred_target: np.ndarray,
    pred_hat: np.ndarray,
    gt: np.ndarray,
    probe_gt_std: np.ndarray,
    names: Tuple[str, ...] = NAMES,
) -> None:
    header = (
        f"{'split':8} {'n':>6} {'dim':>4} "
        f"{'tgt R2':>8} {'hat R2':>8} {'dR2':>8} "
        f"{'tgt MSE':>8} {'hat MSE':>8} {'hat Pixel RMSE':>12} "
        f"{'tgt Pixel RMSE':>12}"
    )
    print(header)
    print("-" * len(header))
    for split, mask in splits.items():
        r2_t, mse_t, pixel_rmse_t = r2_mse(pred_target[mask], gt[mask], probe_gt_std)
        r2_h, mse_h, pixel_rmse_h = r2_mse(pred_hat[mask], gt[mask], probe_gt_std)
        n = int(mask.sum())
        for i, name in enumerate(names):
            print(
                f"{split:8} {n:6d} {name:>4} "
                f"{r2_t[i]:8.4f} {r2_h[i]:8.4f} {r2_h[i] - r2_t[i]:8.4f} "
                f"{mse_t[i]:8.4f} {mse_h[i]:8.4f} {pixel_rmse_h[i]:12.4f} {pixel_rmse_t[i]:12.4f}"
            )
        print("-" * len(header))


def fit_ridge_regression(
    z: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a multi-output linear ridge-regression probe.

    The resulting prediction is:

        y_hat = z @ beta + bias

    Args:
        z: Encoder features with shape (N, D), where:
            N = number of observations
            D = number of latent features
        y: Regression targets with shape (N, K), where:
            K = number of target variables, currently 4
        alpha: Ridge regularization strength. Larger values shrink beta
            more strongly and improve stability in low-variance or
            highly correlated feature directions.

    Returns:
        beta: Feature coefficients with shape (D, K).
        bias: Intercept with shape (K,).
    """

    # Compute the mean value of every feature.
    # Shape: (1, D). Keeping the first dimension enables broadcasting.
    z_mean = z.mean(dim=0, keepdim=True)

    # Compute the mean value of every target.
    # Shape: (1, K).
    gt_mean = gt.mean(dim=0, keepdim=True)

    # Center each feature around zero.
    # Shape: (N, D).
    z_centered = z - z_mean

    # Center each target around zero.
    # Shape: (N, K).
    y_centered = gt - gt_mean

    # Number of observations used to estimate covariance.
    n = z.shape[0]

    # Number of latent features.
    num_features = z.shape[1]

    # Identity matrix used to add alpha to every eigenvalue of the
    # feature covariance matrix.
    # Shape: (D, D).
    identity = torch.eye(
        num_features,
        device=z.device,
        dtype=z.dtype,
    )

    # Feature covariance/Gram matrix:
    #
    #     gram[i, j] = mean(z_i * z_j)
    #
    # Diagonal entries are feature variances.
    # Off-diagonal entries are feature covariances.
    #
    # Shape:
    #     (D, N) @ (N, D) -> (D, D)
    #
    # Division by n gives the population covariance convention.
    covariance = z_centered.T @ z_centered / n

    # Cross-covariance between latent features and targets:
    #
    #     cross[i, j] = mean(z_feature_i * y_target_j)
    #
    # It measures how strongly each feature varies with each target.
    #
    # Shape:
    #     (D, N) @ (N, K) -> (D, K)
    cross_covariance = z_centered.T @ y_centered / n

    # Solve the ridge-regression normal equations:
    #
    #     (covariance + alpha * I) @ beta = cross_covariance
    #
    # This is mathematically equivalent to:
    #
    #     beta = inverse(covariance + alpha * I) @ cross_covariance
    #
    # but solve() is more efficient and numerically stable than
    # explicitly constructing the inverse.
    #
    # In covariance's eigenvector basis, each component is divided by
    # (eigenvalue + alpha). Alpha therefore prevents very small
    # eigenvalues from producing unstable, excessively large weights.
    #
    # Shapes:
    #     (D, D) solving against (D, K) -> beta (D, K)
    beta = torch.linalg.solve(
        covariance + alpha * identity,
        cross_covariance,
    )

    # Centering removed the intercept while fitting beta. Recover it so
    # the prediction at the original feature mean equals the target mean:
    #
    #     y_mean = z_mean @ beta + bias
    #     bias = y_mean - z_mean @ beta
    #
    # Shape: (K,).
    bias = gt_mean.squeeze(0) - z_mean.squeeze(0) @ beta

    return beta, bias
