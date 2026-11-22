"""Cross-entropy method (CEM) over action sequences.

Knows nothing about models or the simulator: it searches with whatever `cost_fn` it is
given, so the model and oracle planners run the same search.
"""

from collections.abc import Callable

import torch

import wm.env.sim as sim

CEM_ITERS = 4
CEM_SAMPLES = 128  # per iteration; 4 x 128 = 512, the random-shooting budget
CEM_ELITE_FRAC = 0.1
CEM_INIT_STD = 1.0
CEM_MIN_STD = 0.05  # keeps the search from collapsing onto one plan

CostFn = Callable[[torch.Tensor], torch.Tensor]  # (N, H, 2) -> (N,)


def sample_actions(
    means: torch.Tensor, stds: torch.Tensor, n: int, rng: torch.Generator
) -> torch.Tensor:
    """Draw n plans from independent Gaussians per (h, axis), clipped to the action bounds.

    (H, 2), (H, 2) -> (n, H, 2)
    """
    lo, hi = sim.ACTION_BOUNDS
    means = means.expand(n, *means.shape)
    stds = stds.expand(n, *stds.shape)
    return torch.normal(means, stds, generator=rng).clamp(lo, hi)


def sort_and_keep_elites(
    actions: torch.Tensor, costs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The lowest-cost fraction of the plans and their costs: -> (K, H, 2), (K,)."""
    k = max(2, int(len(costs) * CEM_ELITE_FRAC))  # std needs at least 2
    top = torch.topk(costs, k, largest=False)
    return actions[top.indices], top.values


def update_distributions(elites: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of the elites, (K, H, 2) -> (H, 2) each; std floored at CEM_MIN_STD."""
    return elites.mean(dim=0), elites.std(dim=0).clamp_min(CEM_MIN_STD)


def cem(
    cost_fn: CostFn,
    init_means: torch.Tensor,
    rng: torch.Generator,
    n_iters: int = CEM_ITERS,
    init_std: float = CEM_INIT_STD,
    history: list[dict] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimise cost_fn over plans, starting from N(init_means, init_std) per (h, axis).

    Returns the final means (H, 2), the plan to execute; the final stds (H, 2), how far the
    search narrowed; and the last iteration's samples (CEM_SAMPLES, H, 2), so the caller can
    score them for diagnostics. If `history` is a list, one dict per iteration is appended
    (for convergence checks).
    """
    means = init_means
    stds = torch.full_like(init_means, init_std)
    for i in range(n_iters):
        actions = sample_actions(means, stds, CEM_SAMPLES, rng)
        costs = cost_fn(actions)
        elites, elite_costs = sort_and_keep_elites(actions, costs)
        means, stds = update_distributions(elites)
        if history is not None:
            history.append(
                {
                    "iter": i + 1,
                    "best_cost": costs.min().item(),
                    "elite_cost": elite_costs.mean().item(),
                    "std": stds.mean().item(),
                    "means": means.clone(),
                }
            )
    return means, stds, actions
