# World models

Research code for a JEPA-style world model on a toy 2D ball. The encoder reads 128×128 frames, a predictor learns the effect of a 2D force in latent space, and a planner uses that model to drive the ball to a target. The write-up is elsewhere: [pixels, collapse, and probes](https://jackholt44.github.io/2026/11/16/world-model-part-1-pixels-collapse-probes/) (velocity probe R² 0.45 against 0.93 recoverable from the pixels, then 0.79 after rebuilding the encoder), [rollouts](https://jackholt44.github.io/2026/11/19/world-model-part-2-rollouts/) (step-5 R² 0.95, then overconfident past the training horizon), and [planning](https://jackholt44.github.io/2026/11/21/world-model-part-3-planning/) (CEM settles about 0.9 px from the target, 0.56 px behind a perfect-model oracle). Research code; see the posts for results.

## Run

Python 3.13 is pinned (3.11+ is the minimum). [uv](https://docs.astral.sh/uv/) installs from the lockfile. Training uses the Apple GPU when PyTorch MPS is available, otherwise CPU.

```bash
uv sync
uv run python -m wm.env              # 10,000 episodes → src/wm/env/data/episodes_v2.npz
uv run python -m wm.train.rollout    # checkpoint under src/wm/train/runs/<timestamp>/
uv run python -m wm.probe            # probes on the latest run
uv run python -m wm.plan             # 50-seed model vs oracle sweep
uv run pytest
```

The generator default matches the later posts: 10,000 episodes of 20 steps, seed 0, with the first 1,000 held out as the test set. `--episodes` and `--n-train` shrink that for a shorter run. Training defaults to a horizon of 5 and 200 epochs. `wm.plan` is the full sweep and expects a checkpoint at `src/wm/train/runs/latest`.

Datasets, checkpoints, and logs are not in this repo. The commands above regenerate them. `src/wm/train/rollout/rollout.ipynb` and `src/wm/plan/mpc.ipynb` read those local runs; the figures saved in the notebooks are from the runs behind the posts.

[MIT](LICENSE) licensed.
