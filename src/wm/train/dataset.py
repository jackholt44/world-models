from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import torch

from wm.env.paths import EPISODES_NPZ

# the first N_TEST_EPISODES episodes are the test set for every run, so data sweeps
# that train on more or fewer episodes are always scored on the same episodes
N_TEST_EPISODES = 1000


class AmiBatch(NamedTuple):
    obs_tminus1: torch.Tensor
    obs_t: torch.Tensor
    obs_tp1: torch.Tensor
    action: torch.Tensor
    gt_state_tminus1: torch.Tensor
    gt_state_t: torch.Tensor
    gt_state_tp1: torch.Tensor
    episode_id: torch.Tensor
    t: torch.Tensor


class Episodes(NamedTuple):
    """Episode-major storage: every frame is stored once.

    An episode of n_steps actions has n_steps + 1 frames. Transition k (1 <= k < n_steps)
    is (frame k-1, frame k) -> action k -> (frame k, frame k+1); frame 0 has no previous
    frame, so it is only ever used as obs_tminus1.
    """

    obs: torch.Tensor  # (E, n_steps + 1, 3, H, W) uint8
    action: torch.Tensor  # (E, n_steps, 2) float32
    gt_state: torch.Tensor  # (E, n_steps + 1, 4) float32
    episode_id: np.ndarray  # (E,) str

    @property
    def n_steps(self) -> int:
        return self.action.shape[1]

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, index: slice) -> "Episodes":
        return Episodes(
            obs=self.obs[index],
            action=self.action[index],
            gt_state=self.gt_state[index],
            episode_id=self.episode_id[index],
        )


def load_episodes(path: Path = EPISODES_NPZ) -> Episodes:
    # the generator already stores CHW uint8, so from_numpy wraps the arrays without a copy
    with np.load(path) as data:
        return Episodes(
            obs=torch.from_numpy(data["obs"]),
            action=torch.from_numpy(data["action"]),
            gt_state=torch.from_numpy(data["gt_state"]),
            episode_id=np.asarray(data["episode_id"]),
        )


def split_episodes(
    episodes: Episodes, n_test: int = N_TEST_EPISODES, n_train: int | None = None
) -> Tuple[Episodes, Episodes]:
    """Split into (train, test) by episode.

    Episodes are generated i.i.d., so contiguous slices are a fair split and are views,
    not copies. Test is always the first n_test episodes; train is the next n_train
    (all of the rest by default), so smaller train sets are nested in larger ones.
    """
    if n_train is None:
        n_train = len(episodes) - n_test
    if n_test + n_train > len(episodes):
        raise ValueError(
            f"Asked for {n_test} test + {n_train} train episodes but only {len(episodes)} exist"
        )
    return episodes[n_test : n_test + n_train], episodes[:n_test]


def _transitions(
    episodes: Episodes, episode_index: int, step_index: int, horizon: int | None
) -> AmiBatch:
    """Transition k of episode e, or the window of `horizon` transitions starting at k.

    With horizon=None the fields have no time axis (one transition); otherwise every
    field except episode_id has a leading time axis of length horizon.
    """
    if horizon is None:
        prev, cur, nxt = step_index - 1, step_index, step_index + 1
        t = torch.tensor(step_index)
    else:
        prev, cur, nxt = (
            slice(step_index + offset, step_index + offset + horizon) for offset in (-1, 0, 1)
        )
        t = torch.arange(step_index, step_index + horizon)

    return AmiBatch(
        obs_tminus1=episodes.obs[episode_index, prev],
        obs_t=episodes.obs[episode_index, cur],
        obs_tp1=episodes.obs[episode_index, nxt],
        action=episodes.action[episode_index, cur],
        gt_state_tminus1=episodes.gt_state[episode_index, prev],
        gt_state_t=episodes.gt_state[episode_index, cur],
        gt_state_tp1=episodes.gt_state[episode_index, nxt],
        episode_id=torch.tensor(episode_index),
        t=t,
    )


class AmiDataset(torch.utils.data.Dataset):
    """One item per transition, with no time axis."""

    def __init__(self, episodes: Episodes):
        self.episodes = episodes
        self.transitions_per_episode = episodes.n_steps - 1

    def __len__(self):
        return len(self.episodes) * self.transitions_per_episode

    # this is called by the dataloader to get a batch of data
    def __getitem__(self, index):
        episode_index, step_index = divmod(index, self.transitions_per_episode)
        # step_index + 1 because the first frame has no previous frame
        return _transitions(self.episodes, episode_index, step_index + 1, horizon=None)


class RolloutDataset(torch.utils.data.Dataset):
    """One item per window of `horizon` consecutive transitions within one episode."""

    def __init__(self, episodes: Episodes, horizon: int = 5):
        if not 1 <= horizon <= episodes.n_steps - 1:
            raise ValueError(f"horizon must be in [1, {episodes.n_steps - 1}], got {horizon}")
        self.episodes = episodes
        self.horizon = horizon
        # windows start at k = 1 .. n_steps - horizon
        self.windows_per_episode = episodes.n_steps - horizon

    def __len__(self):
        return len(self.episodes) * self.windows_per_episode

    def __getitem__(self, index):
        e, k = divmod(index, self.windows_per_episode)
        return _transitions(self.episodes, e, k + 1, horizon=self.horizon)


class RolloutBatch(NamedTuple):
    frames: torch.Tensor  # (B, horizon + 2, 3, H, W) uint8: frames k-1 .. k+horizon
    action: torch.Tensor  # (B, horizon, 2)


class RolloutBatches:
    """Batches of the same windows as RolloutDataset, built for training speed.

    Each batch is gathered with one indexing op instead of batch_size __getitem__ calls
    plus collate, and carries the horizon + 2 distinct frames of each window rather than
    the 3 * horizon overlapping frames in obs_tminus1/obs_t/obs_tp1, which halves the
    host-to-device copy. Like a shuffled DataLoader, every window appears once per epoch.
    """

    def __init__(
        self,
        episodes: Episodes,
        horizon: int = 5,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not 1 <= horizon <= episodes.n_steps - 1:
            raise ValueError(f"horizon must be in [1, {episodes.n_steps - 1}], got {horizon}")
        self.episodes = episodes
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        # windows start at k = 1 .. n_steps - horizon, as in RolloutDataset
        self.windows_per_episode = episodes.n_steps - horizon
        self.n_windows = len(episodes) * self.windows_per_episode

    def __len__(self):
        return -(-self.n_windows // self.batch_size)

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n_windows, generator=self.generator)
        else:
            order = torch.arange(self.n_windows)
        frame_offsets = torch.arange(self.horizon + 2)
        action_offsets = torch.arange(self.horizon)
        for index in order.split(self.batch_size):
            episode_index = index // self.windows_per_episode
            step_index = index % self.windows_per_episode + 1
            yield RolloutBatch(
                frames=self.episodes.obs[
                    episode_index[:, None], (step_index - 1)[:, None] + frame_offsets
                ],
                action=self.episodes.action[
                    episode_index[:, None], step_index[:, None] + action_offsets
                ],
            )


class EpisodeDataset(RolloutDataset):
    """One item per episode: every transition of the episode, with a leading time axis."""

    def __init__(self, episodes: Episodes):
        super().__init__(episodes, horizon=episodes.n_steps - 1)
