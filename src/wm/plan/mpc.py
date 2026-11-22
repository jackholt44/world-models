import json
import random
from datetime import datetime
from typing import Literal

import numpy as np
import torch

import wm.env.sim as sim
from wm.env.paths import LATEST_RUN, MPC_LOG
from wm.env.physics import ACTION_SCALE, center_bounds
from wm.nets.inputs import encode_pair, rollout_latent
from wm.plan.cem import CEM_INIT_STD, CEM_ITERS, cem
from wm.probe.runner import LoadedRun, load_run

N_ACTIONS = 512  # evaluations per step: random-shooting candidates, or CEM iters x samples
H = 5
N_CONTROL_STEPS = 50

Planner = Literal["model", "oracle"]
Search = Literal["random", "cem"]


def generate_target(device: torch.device, rng: torch.Generator) -> torch.Tensor:
    """A uniform (x, y) inside the walls, (2,) on `device`; `rng` must live on `device`."""
    x0, x1, y0, y1 = center_bounds()
    low = torch.tensor([x0, y0], device=device)
    high = torch.tensor([x1, y1], device=device)
    u = torch.empty(2, device=device).uniform_(0, 1, generator=rng)
    return low + u * (high - low)


def sample_actions(device: torch.device, rng: torch.Generator) -> torch.Tensor:
    lo, hi = sim.ACTION_BOUNDS
    return torch.empty(N_ACTIONS, H, 2, device=device).uniform_(lo, hi, generator=rng)


def calculate_position_error(mpc_states: torch.Tensor, sim_states: torch.Tensor) -> torch.Tensor:
    """Calculate the position error between the MPC states and the simulator states."""
    mpc_positions = mpc_states[:, :, :2]
    sim_positions = sim_states[:, :, :2]
    return torch.linalg.norm(mpc_positions - sim_positions, dim=-1).mean(dim=0)  # (N,H,1) -> (H,1)


def stopping_offset(velocity: torch.Tensor, action_scale: float = ACTION_SCALE) -> torch.Tensor:
    """Calculate the stopping offset of the velocity."""
    speed = velocity.abs()
    m = torch.floor(speed / action_scale)
    return torch.sign(velocity) * (m * speed - action_scale * m * (m + 1) / 2)


def calculate_stopping_cost(states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distance from target to where each state would stop under max braking: (..., 4) -> (...).

    At rest the stop point is the position itself, so this already includes the position miss.
    """
    stopped_positions = states[..., :2] + stopping_offset(states[..., 2:])
    return torch.linalg.norm(stopped_positions - target, dim=-1)


def calculate_stopping_costs(states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Running cost: the per-state cost averaged over the horizon, (N, H, 4) -> (N,)."""
    return calculate_stopping_cost(states, target).mean(dim=1)


def calculate_latent_distance(
    trajectories: torch.Tensor, zt: torch.Tensor, loss_type: Literal["l1", "l2"] = "l2"
) -> torch.Tensor:
    """Calculate the latent distance between the trajectories and the target.
    (N, H, z) -> (N, H)
    """
    d = trajectories - zt
    if loss_type == "l1":
        return d.abs_().sum(dim=-1)
    elif loss_type == "l2":
        return torch.linalg.norm(d, dim=-1)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


@torch.no_grad()
def model_costs(
    run: LoadedRun,
    frame_prev: torch.Tensor,
    frame_now: torch.Tensor,
    actions: torch.Tensor,
    zt: torch.Tensor,
    loss_type: Literal["l1", "l2"] = "l2",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latent cost of each sequence, (N, H, 2) -> (N,), its probe-decoded states (N, H, 4),
    and the latent trajectories (N, H, z).

    The states are for diagnostics only (position error against the sim); the cost never
    uses the probe.
    """
    nets = run.nets
    z0 = encode_pair(nets.encoder, frame_prev, frame_now, run.device)  # (z,)
    trajectories = rollout_latent(nets.predictor, z0.expand(len(actions), -1), actions)  # (N, H, z)
    distances = calculate_latent_distance(trajectories, zt, loss_type)
    costs = distances.mean(dim=1)  # (N, H) -> (N,)
    states = run.decode_state(trajectories)  # (N, H, 4), logging only
    return costs, states, trajectories


def sim_costs(
    state: np.ndarray, actions: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """True cost of each sequence, as in model_costs: (N, H, 2) -> (N,) on target's device."""
    states = sim.rollout(tuple(state), actions.cpu().numpy())  # (N, H, 4)
    states = torch.as_tensor(states, device=target.device, dtype=target.dtype)
    final_positions = states[:, -1, :2]  # (N, 2)
    costs = calculate_stopping_costs(states, target)
    return costs, final_positions, states


def execute_action(state: np.ndarray, action: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    """Step the simulator. Returns the new state and the uint8 CPU frame of it.

    `state` is a host float64 array (x, y, vx, vy). `action` is a length-2 tensor
    on any device; physics only accepts Python floats, so it is copied off device here.
    """
    host_action = tuple(float(v) for v in action.detach().cpu().tolist())
    next_state = np.asarray(sim.step(state, host_action), dtype=np.float64)
    return next_state, frame_tensor(next_state)


def frame_tensor(state: np.ndarray) -> torch.Tensor:
    """uint8 CPU frame, (3, H, W). encode_pair moves it onto the model device."""
    return torch.from_numpy(sim.observe(state))


def is_wall_bounce(
    x_now: float, x_next: float, vx_next: float, vy_next: float, y_now: float, y_next: float
) -> bool:
    """True when position did not follow x += vx * dt or y += vy * dt (dt=1, v-then-x integrator).
    Physics step is v-then-x integrator, so the next position is x_next = x_now + vx_next.
    """

    res_x = x_next - x_now - vx_next
    res_y = y_next - y_now - vy_next

    return (abs(res_x) > 1e-2) | (abs(res_y) > 1e-2)


def calculate_per_horizon_position_error(
    mpc_states: torch.Tensor, sim_states: torch.Tensor
) -> torch.Tensor:
    """Calculate the per-horizon position error between the MPC states and the simulator states."""
    mpc_positions = mpc_states[:, :, :2]
    sim_positions = sim_states[:, :, :2]
    return torch.linalg.norm(mpc_positions - sim_positions, dim=-1).mean(dim=0)  # (N,H,1) -> (H,1)


def choose_action(
    planner: Planner, pred_costs: torch.Tensor, true_costs: torch.Tensor
) -> torch.Tensor:
    """Choose the action with the lowest cost."""
    return pred_costs.argmin() if planner == "model" else true_costs.argmin()


def mpc(
    seed: int = 0,
    planner: Planner = "model",
    run_id: str | None = None,
    loss_type: Literal["l1", "l2"] = "l2",
    search: Search = "cem",
    init_std: float = CEM_INIT_STD,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run closed-loop MPC for one seed and return the final distance to the target.

    Each step searches for a plan with the planner's own cost (latent cost for "model",
    simulator cost for "oracle"), using `search` with the same budget of N_ACTIONS
    evaluations: random shooting, or CEM warm-started from the previous step's plan.
    Then the chosen plan (row 0) and the search's final samples are scored once by both
    the model and the simulator, and per-step logs show:
      pred    — the model's cost of the chosen plan (latent units)
      true    — the simulator's cost of the chosen plan (px)
      best    — the best simulator cost among the plan and the final samples
      regret  — true - best: how much worse the choice is than the best seen
    Under CEM each planner's samples follow its own cost, so model and oracle no longer
    score identical candidates; best and regret are relative to the planner's own samples.

    Three homes, and they stay separate:
      state — numpy float64 (4,) on the host, the only thing sim.step accepts
      frames — uint8 CPU tensors; encode_pair moves them onto run.device
      target, actions, costs — float tensors on run.device
    """

    run = load_run(LATEST_RUN)  # best.pth + its MLP probe
    run.nets.eval()
    sim_rng = random.Random(seed)  # start state + warm-up action
    mpc_rng = torch.Generator(device=run.device).manual_seed(seed)  # target + candidates

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    config = {
        "run_id": run_id,
        "H": H,
        "N": N_ACTIONS,
        "cost": "latent_running",
        "lam": 0.0,
        "loss_type": loss_type,
        "search": search,
        "cem_iters": CEM_ITERS if search == "cem" else None,
        "cem_init_std": init_std if search == "cem" else None,
    }

    state = np.asarray(sim.reset(sim_rng), dtype=np.float64)
    target = generate_target(run.device, mpc_rng)  # (2,) on run.device
    target_frame = frame_tensor(target)
    with torch.no_grad():  # ball at rest on the target, in the predictor's (target) space
        zt = encode_pair(run.nets.encoder_target, target_frame, target_frame, run.device)
    frame_prev = frame_tensor(state)
    state_prev = state  # the state behind frame_prev, for bounce_in
    state = np.asarray(sim.step(state, sim.random_action(sim_rng)), dtype=np.float64)
    frame_now = frame_tensor(state)

    per_step_position_error = np.zeros((N_CONTROL_STEPS, H), dtype=np.float64)
    per_step_regret = np.zeros(N_CONTROL_STEPS, dtype=np.float64)

    plan = torch.zeros(H, 2, device=run.device)  # CEM warm start
    for t in range(N_CONTROL_STEPS):

        def cost_fn(actions: torch.Tensor) -> torch.Tensor:
            if planner == "model":
                return model_costs(run, frame_prev, frame_now, actions, zt, loss_type)[0]
            return sim_costs(state, actions, target)[0]

        # search with the planner's own cost
        if search == "cem":
            warm = torch.cat([plan[1:], torch.zeros(1, 2, device=run.device)])  # shift one step
            plan, plan_std, samples = cem(cost_fn, warm, mpc_rng, init_std=init_std)
        else:
            samples = sample_actions(run.device, mpc_rng)  # (N_ACTIONS, H, 2)
            plan = samples[cost_fn(samples).argmin()]
            plan_std = None

        # diagnostics: score the chosen plan (row 0) and the final samples with both
        plans = torch.cat([plan.unsqueeze(0), samples])  # (1 + N, H, 2)
        pred_costs, mpc_states, _ = model_costs(run, frame_prev, frame_now, plans, zt, loss_type)
        true_costs, _, sim_states = sim_costs(state, plans, target)
        pred_cost, true_cost, best = pred_costs[0], true_costs[0], true_costs.min()
        regret = true_cost - best
        sim_vs_pred = pred_cost - true_cost  # mixed units under the latent cost
        per_horizon_position_error = calculate_per_horizon_position_error(
            mpc_states[1:], sim_states[1:]
        )
        # position error of the chosen plan, to compare with the mean (selection bias)
        chosen_position_error = torch.linalg.norm(
            mpc_states[0, :, :2] - sim_states[0, :, :2], dim=-1
        )  # (H,)

        per_step_position_error[t] = per_horizon_position_error.cpu().numpy()
        dist = np.linalg.norm(state[:2] - target.tolist())
        per_step_regret[t] = regret

        next_state, next_frame = execute_action(state, plan[0])
        is_bounced = is_wall_bounce(
            x_now=state[0],
            x_next=next_state[0],
            vx_next=next_state[2],
            vy_next=next_state[3],
            y_now=state[1],
            y_next=next_state[1],
        )
        # wall hit inside the encoder's input pair (t-1, t): velocity is aliased
        bounce_in = is_wall_bounce(
            x_now=state_prev[0],
            x_next=state[0],
            vx_next=state[2],
            vy_next=state[3],
            y_now=state_prev[1],
            y_next=state[1],
        )

        print(
            f"t={t:2d}  dist={dist:6.2f}  pred={pred_cost:6.2f}  "
            f"true={true_cost:6.2f}  "
            f"best={best:6.2f}  regret={regret:5.2f}  bounced={is_bounced}  "
            f"sim_vs_pred={sim_vs_pred:5.2f}"
        )

        log = {
            **config,
            "seed": seed,
            "step": t,
            "planner": planner,
            "dist": dist,
            "speed": float(np.hypot(state[2], state[3])),
            "pred": pred_cost,
            "true": true_cost,
            "best": best,
            "regret": regret,
            "bounced": is_bounced,
            "bounce_in": bounce_in,
            "per_horizon_position_error": per_horizon_position_error.cpu().numpy(),
            "chosen_position_error": chosen_position_error.cpu().numpy(),
            "plan": plan.cpu().numpy(),  # (H, 2) executed plan; |a| near 1 = saturated
            "plan_std": None if plan_std is None else plan_std.cpu().numpy(),  # (H, 2)
        }
        frame_prev = frame_now
        frame_now = next_frame
        state_prev = state
        state = next_state

        MPC_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(MPC_LOG, "a") as f:
            f.write(json.dumps(log, default=lambda o: o.tolist()) + "\n")

    final_dist = float(np.linalg.norm(state[:2] - target.tolist()))
    # each row above holds the state before its action; this row holds the state after the
    # last one, so success (final dist and speed) can be read from the log
    final_log = {
        **config,
        "seed": seed,
        "step": N_CONTROL_STEPS,
        "planner": planner,
        "final": True,
        "dist": final_dist,
        "speed": float(np.hypot(state[2], state[3])),
    }
    with open(MPC_LOG, "a") as f:
        f.write(json.dumps(final_log) + "\n")
    print(f"[{planner}] final state: {state}  final dist: {final_dist:.2f}")
    return final_dist, per_step_position_error, per_step_regret


if __name__ == "__main__":
    mpc()
