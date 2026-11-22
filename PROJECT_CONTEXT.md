## World Model Side Project — Project Context (Light)

### One-sentence goal
Build a minimal JEPA-style **latent dynamics model** (System A) and a simple **planner** (System B) that uses it to move a ball to a target, with explicit diagnostics for collapse and model-bias failures.

### Non-goals (to prevent scope drift)
- We are **not** building a high-fidelity physics simulator.
- We are **not** choosing between robotics/real-world physics engines (e.g., Matter/Planck) for realism.
- We are **not** optimizing for transfer to real tasks yet; we’re optimizing for **learning the model failure modes** (collapse, compounding rollout error, planner mismatch).
### Phase scope (defaults)
- **Phase 1**: a **toy** 2D sim: single ball bouncing off walls (walls only, **no collisions** with other objects).
- **Observations**: RGB frames, default `128x128` (can change later, but keep fixed per experiment).
- **Actions**: 2D force vector applied to the ball each step, default range `[-1, 1]` then scaled.
- **Ground truth state** (debug only): position `(x,y)`, velocity `(vx,vy)` in continuous units.

### Why we start with pixels anyway (despite state being “easier”)
- Pixels make a clean end-to-end demo of **representation learning** + dynamics in latent space.
- Ground-truth state is still logged purely for **debugging and probes**, not as the primary training target.

### What we mean by “JEPA-style” here
We are *not* predicting pixels. We learn an encoder \(e\) that maps a frame to a latent \(z_t = e(x_t)\), and a predictor \(p\) that predicts the next latent:
- **Passive**: \(\hat z_{t+1} = p(z_t)\)
- **Action-conditioned**: \(\hat z_{t+1} = p(z_t, a_t)\)

Training target is produced by a **separate target encoder** \(e_{ema}\) updated by EMA:
\(z^*_{t+1} = e_{ema}(x_{t+1})\)

Primary objective: make \(\hat z_{t+1}\) close to \(z^*_{t+1}\) (plus anti-collapse regularization).

### Dataset record (minimum fields)
Each step contributes one training example with:
- `obs_t` (RGB frame)
- `action_t` (2 floats)
- `obs_tp1` (RGB frame at next step)
- `gt_state_t`, `gt_state_tp1` (pos/vel) for debugging + probes
- `episode_id`, `t` (for sequential eval / rollouts)

### Core metrics to log
- **Collapse checks**:
  - latent per-dimension variance (should not go ~0)
  - covariance off-diagonal magnitude (should not explode)
  - pairwise cosine similarity distribution across batch (should not concentrate at 1)
- **Dynamics checks**:
  - 1-step latent prediction error
  - H-step rollout error as a function of horizon H (error-vs-H curve)
- **Planning checks**:
  - goal success rate
  - calibration gap: predicted distance-to-goal vs realized distance-to-goal after executing first action

### Failure taxonomy (labels we use in discussion)
- **Representation collapse**: encoder outputs near-constant latents; loss can look good but planning fails.
- **Transition error**: 1-step is bad; model doesn’t learn dynamics.
- **Compounding error**: 1-step ok, H-step bad; small bias explodes over rollouts.
- **Planner distribution shift**: planner picks actions outside training distribution; model is wrong where it’s used.

### “expected_failure” (important clarification)
`expected_failure` is **not** a dataset label. It’s a field in your **experiment log**: a short prediction of what you think will go wrong and how you’ll detect it (e.g., “collapse by epoch 5; variance -> 0”).

### Minimal planner definition (MPC loop)
At each env step:
1. sample N candidate action sequences of horizon H
2. rollout in world model: \(z_{t+k+1} = p(z_{t+k}, a_{t+k})\)
3. score final latent using a cost (via probe/decoder to position or a latent-cost proxy)
4. execute only the first action in the real env, observe, re-encode, repeat

### Evidence we need for “latent encodes physics”
- A simple probe (linear/MLP) from \(z_t\) to `(x,y,vx,vy)` that generalizes across episodes.
- Visual sanity checks (e.g., latent similarity correlates with proximity in state space more than color/noise).

