from datetime import datetime

import numpy as np

from wm.env.paths import MPC_LOG
from wm.plan.mpc import N_CONTROL_STEPS, H, mpc

N_SEEDS = 50
SEEDS = range(N_SEEDS)
LOSS_TYPE = "l2"
INIT_STD = 0.5  # CEM starting std per step


def main():
    # rows append to MPC_LOG; filter on run_id to read back one sweep
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"run_id: {run_id}  log: {MPC_LOG}")

    model_distances = np.zeros(N_SEEDS, dtype=np.float64)
    true_distances = np.zeros(len(SEEDS), dtype=np.float64)
    per_step_position_errors = np.zeros((N_SEEDS, N_CONTROL_STEPS, H), dtype=np.float64)
    per_step_position_errors_sim = np.zeros((N_SEEDS, N_CONTROL_STEPS, H), dtype=np.float64)
    per_step_regrets = np.zeros((N_SEEDS, N_CONTROL_STEPS), dtype=np.float64)
    per_step_regrets_sim = np.zeros((N_SEEDS, N_CONTROL_STEPS), dtype=np.float64)
    for seed in SEEDS:
        model_distance, per_step_per_horizon_position_error_mpc, per_step_regret_mpc = mpc(
            seed, planner="model", run_id=run_id, loss_type=LOSS_TYPE, init_std=INIT_STD
        )
        true_distance, per_step_per_horizon_position_error_sim, per_step_regret_sim = mpc(
            seed, planner="oracle", run_id=run_id, loss_type=LOSS_TYPE, init_std=INIT_STD
        )

        model_distances[seed] = model_distance
        true_distances[seed] = true_distance
        per_step_position_errors[seed] = per_step_per_horizon_position_error_mpc
        per_step_position_errors_sim[seed] = per_step_per_horizon_position_error_sim
        per_step_regrets[seed] = per_step_regret_mpc
        per_step_regrets_sim[seed] = per_step_regret_sim

    avg_position_error_mpc = per_step_position_errors[:, 30:50, -1].mean(axis=(0))
    avg_position_error_sim = per_step_position_errors_sim[:, 30:50, -1].mean(axis=(0))
    avg_regret_mpc = per_step_regrets[:, 30:50].mean(axis=(0))
    avg_regret_sim = per_step_regrets_sim[:, 30:50].mean(axis=(0))

    print(
        f"Mean Final Distance MPC: {model_distances.mean():.2f}, Std: {model_distances.std():.2f}"
    )
    print(f"Mean Final Distance Sim: {true_distances.mean():.2f}, Std: {true_distances.std():.2f}")

    print("Average Position Error")
    for i in range(len(avg_position_error_mpc)):
        print(
            f"Step {31 + i}: MPC {avg_position_error_mpc[i]:.2f}, "
            f"Sim {avg_position_error_sim[i]:.2f}"
        )

    print("Average Regret")
    for i in range(len(avg_regret_mpc)):
        print(f"Step {31 + i}: MPC {avg_regret_mpc[i]:.2f}, Sim {avg_regret_sim[i]:.2f}")


if __name__ == "__main__":
    main()
