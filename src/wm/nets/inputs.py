"""Stored frames -> encoder inputs, and the predictor rolled forward on its own outputs.

Training, evaluation and planning all go through these, so they can't disagree about
pixel scaling, frame order or how a rollout is stepped.
"""

import torch


def frames_to_float(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 frames in [0, 255] -> float32 in [0, 1] on `device`, any leading shape."""
    if frames.dtype != torch.uint8:
        # div_ is in place, so a float input already on `device` would be scaled twice
        raise TypeError(f"expected uint8 frames, got {frames.dtype}")
    # copy uint8 and cast on the device: casting first would copy 4x the bytes
    return frames.to(device).float().div_(255.0)


def frame_pair(frame_prev: torch.Tensor, frame_now: torch.Tensor) -> torch.Tensor:
    """(..., 3, H, W) x 2 -> (..., 6, H, W), the earlier frame in the first 3 channels."""
    return torch.cat([frame_prev, frame_now], dim=-3)


def encode_pair(
    encoder: torch.nn.Module,
    frame_prev: torch.Tensor,
    frame_now: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode uint8 frame pairs: (..., 3, H, W) x 2 -> (..., latent).

    Leading dims (e.g. batch and time) are flattened for the encoder and restored after.
    """
    pair = frame_pair(frames_to_float(frame_prev, device), frames_to_float(frame_now, device))
    lead = pair.shape[:-3]
    z = encoder(pair.reshape(-1, *pair.shape[-3:]))
    return z.reshape(*lead, z.shape[-1])


def rollout_latent(
    predictor: torch.nn.Module, z0: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Predict H steps from z0 using only the predictor's own outputs.

    (B, latent) and (B, H, 2) -> (B, H, latent); column h is h + 1 steps ahead.
    """
    z = z0
    trajectory = []
    for h in range(actions.shape[1]):
        z = predictor(torch.cat([z, actions[:, h]], dim=-1))
        trajectory.append(z)
    return torch.stack(trajectory, dim=1)
