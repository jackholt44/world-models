from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from wm.env import sim
from wm.env.paths import EPISODES_NPZ
from wm.env.physics import ACTION_SCALE, ARENA_HEIGHT, ARENA_WIDTH, BALL_RADIUS, MASS

NO_EPISODES = 10000
NO_STEPS = 20


def generate_episode(
    rng: random.Random, obs: np.ndarray, action: np.ndarray, gt_state: np.ndarray
) -> None:
    """Simulate one episode, writing into preallocated arrays.

    obs and gt_state have NO_STEPS + 1 rows (every frame, including the one after the
    last action); action has NO_STEPS rows. Writing in place keeps peak memory at one
    copy of the dataset, rather than a list of frames plus the stacked array.
    """
    state = sim.reset(rng)

    for i in range(NO_STEPS):
        obs[i] = sim.observe(state)
        gt_state[i] = state
        a = sim.random_action(rng)
        action[i] = a
        state = sim.step(state, a)

    obs[NO_STEPS] = sim.observe(state)
    gt_state[NO_STEPS] = state


def main(no_episodes: int = NO_EPISODES, out: Path = EPISODES_NPZ) -> None:
    seed = 0
    rng = random.Random(seed)

    height, width = int(ARENA_HEIGHT), int(ARENA_WIDTH)
    obs = np.empty((no_episodes, NO_STEPS + 1, 3, height, width), dtype=np.uint8)
    action = np.empty((no_episodes, NO_STEPS, 2), dtype=np.float32)
    gt_state = np.empty((no_episodes, NO_STEPS + 1, 4), dtype=np.float32)
    episode_id = np.empty(no_episodes, dtype="<U36")

    for i in tqdm(range(no_episodes), desc="Generating episodes"):
        episode_id[i] = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"episode/seed_{seed:04d}/episode_{i:04d}")
        )
        generate_episode(rng, obs[i], action[i], gt_state[i])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=obs,
        action=action,
        gt_state=gt_state,
        episode_id=episode_id,
    )

    meta_path = out.with_suffix(".json")
    meta: Dict[str, Any] = {
        "dataset": "ball_episodes",
        "schema_version": 2,
        "seed": seed,
        "no_episodes": no_episodes,
        "no_steps": NO_STEPS,
        "dt": sim.DT,
        "physics": {
            "mass": MASS,
            "action_scale": ACTION_SCALE,
            "arena_width": ARENA_WIDTH,
            "arena_height": ARENA_HEIGHT,
            "ball_radius": BALL_RADIUS,
        },
        "action_stored_as": "normalized_xy_in_minus_one_one",
        "action_note": (
            "npz 'action' rows are (fx, fy) from Uniform(-1, 1). "
            "physics_step scales by ACTION_SCALE before integrating."
        ),
        "layout_note": (
            "Episode-major. action[e, k] moves frame k to frame k+1. "
            "Transitions are built from frames (k-1, k, k+1) in wm.train.dataset."
        ),
        "arrays": {
            "obs_shape": list(obs.shape),
            "action_shape": list(action.shape),
            "gt_state_shape": list(gt_state.shape),
            "dtype_obs": str(obs.dtype),
            "obs_layout": "episode, frame, channel, height, width",
            "dtype_action": str(action.dtype),
            "dtype_gt_state": str(gt_state.dtype),
        },
        "files": {"npz": str(out), "meta": str(meta_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("Saved dataset to", out)
    print("Saved metadata to", meta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NO_EPISODES, help="Number of episodes")
    parser.add_argument("--out", type=Path, default=EPISODES_NPZ, help="Output .npz path")
    args = parser.parse_args()

    main(no_episodes=args.episodes, out=args.out)
