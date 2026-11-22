import json
import sys
from pathlib import Path

import torch


def write_training_rollout_to_file(
    epoch: int,
    per_step_loss: torch.Tensor,
    variance_loss: float | torch.Tensor,
    grad_norm: float | torch.Tensor,
    mse_losses: torch.Tensor,
    r2_scores: torch.Tensor,
    copy_z_mse_losses: torch.Tensor,
    z_hat_variances: torch.Tensor,
    z_target_variances: torch.Tensor,
    log_path: Path,
    resume: bool = False,
) -> None:
    log_entry = {
        "epoch": epoch,
        "train_per_step_loss": per_step_loss.detach().cpu().tolist(),
        "train_variance_loss": float(variance_loss),
        "train_grad_norm": float(grad_norm),
        "test_mse_losses": mse_losses.detach().cpu().tolist(),
        "test_r2_scores": r2_scores.detach().cpu().tolist(),
        "test_copy_z_mse_losses": copy_z_mse_losses.detach().cpu().tolist(),
        "test_z_hat_variances": z_hat_variances.detach().cpu().tolist(),
        "test_z_target_variances": z_target_variances.detach().cpu().tolist(),
    }
    with open(log_path, "w" if not resume and epoch == 0 else "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        f.flush()


_LABEL_W = 11
_VALUE_W = 8
_EP_W = 4


def format_rollout_row(
    epoch: int,
    per_step_loss: torch.Tensor,
    variance_loss: float | torch.Tensor,
    grad_norm: float | torch.Tensor,
    mse_losses: torch.Tensor,
    r2_scores: torch.Tensor,
    copy_z_mse: torch.Tensor,
    z_hat_variances: torch.Tensor,
    z_target_variances: torch.Tensor,
    prev_loss: float | None,
    prev_mse: float | None,
    prev_r2: float | None,
) -> str:
    horizon = int(per_step_loss.shape[0])
    loss_mean = _as_float(per_step_loss.mean())
    mse_mean = _as_float(mse_losses.mean())
    r2_mean = _as_float(r2_scores.mean())
    test = [
        _block(
            "mse",
            mse_mean,
            mse_losses,
            horizon,
            _trend_code(mse_mean, prev_mse, higher_is_better=False),
        ),
        _block(
            "r2",
            r2_mean,
            r2_scores,
            horizon,
            _trend_code(r2_mean, prev_r2, higher_is_better=True),
        ),
        _block("z0_copy_mse", _as_float(copy_z_mse.mean()), copy_z_mse, horizon),
        _block("zhat_var", _as_float(z_hat_variances.mean()), z_hat_variances, horizon),
        _block("ztgt_var", _as_float(z_target_variances.mean()), z_target_variances, horizon),
    ]
    train = [
        _block(
            "train_loss",
            loss_mean,
            per_step_loss,
            horizon,
            _trend_code(loss_mean, prev_loss, higher_is_better=False),
        ),
        _block("hinge_loss", _as_float(variance_loss), None, horizon),
        _block("gnorm", _as_float(grad_norm), None, horizon),
    ]
    rows = _side_by_side(test, train).split("\n")
    epoch_label = f"{epoch:<{_EP_W}}"
    pad = " " * _EP_W
    lines = [f"{epoch_label}{rows[0]}"]
    lines.extend(f"{pad}{row}" for row in rows[1:])
    return "\n".join(lines)


def format_rollout_header(horizon: int) -> str:
    steps = _step_header(horizon)
    return f"{'':<{_EP_W}}{steps}  {steps}"


def _as_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _paint(text: str, code: str | None) -> str:
    if code is None or not sys.stdout.isatty():
        return text
    # Reset first so a leftover dim/bold attribute from a Ctrl+C cannot stick.
    return f"\033[0;{code}m{text}\033[0m"


def _trend_code(current: float, previous: float | None, *, higher_is_better: bool) -> str | None:
    if previous is None or current == previous:
        return None
    improved = current > previous if higher_is_better else current < previous
    return "32" if improved else "31"


def _num(value: float | torch.Tensor) -> str:
    return f"{_as_float(value):>{_VALUE_W}.4f}"


def _block(
    name: str,
    mean: float,
    steps: torch.Tensor | None,
    horizon: int,
    code: str | None = None,
) -> str:
    cells = [f"{name:<{_LABEL_W}}", _paint(_num(mean), code)]
    for index in range(horizon):
        cells.append(_num(steps[index]) if steps is not None else " " * _VALUE_W)
    return " ".join(cells)


def _step_header(horizon: int) -> str:
    steps = " ".join(f"{f's{index + 1}':>{_VALUE_W}}" for index in range(horizon))
    return f"{'':<{_LABEL_W}} {'avg':>{_VALUE_W}} {steps}"


def _side_by_side(left: list[str], right: list[str]) -> str:
    blank = " " * (_LABEL_W + 1 + _VALUE_W)
    lines = []
    for index in range(max(len(left), len(right))):
        lhs = left[index] if index < len(left) else blank
        rhs = right[index] if index < len(right) else ""
        lines.append(f"{lhs}  {rhs}".rstrip())
    return "\n".join(lines)
