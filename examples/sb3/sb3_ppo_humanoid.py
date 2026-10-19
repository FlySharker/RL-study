"""Train and run PPO on MuJoCo Humanoid-v5 using Stable-Baselines3.

Install dependency:
    pip install -r requirements.txt

Train:
    python examples/sb3/sb3_ppo_humanoid.py --device cpu

Train with TensorBoard logging:
    python examples/sb3/sb3_ppo_humanoid.py --device cpu --tensorboard-log logs/tensorboard

Start TensorBoard:
    tensorboard --logdir logs/tensorboard

Evaluate a saved model:
    python examples/sb3/sb3_ppo_humanoid.py --no-train --load-path models/sb3_ppo_humanoid.zip

Watch a saved model:
    python examples/sb3/sb3_ppo_humanoid.py --no-train --load-path models/sb3_ppo_humanoid.zip --watch
"""

from __future__ import annotations

import argparse

from sb3_ppo_mujoco_common import add_common_args, run_ppo_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable-Baselines3 PPO for MuJoCo Humanoid-v5.")
    add_common_args(
        parser,
        env_id="Humanoid-v5",
        total_timesteps=5_000_000,
        n_envs=8,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        use_sde=False,
        sde_sample_freq=-1,
        hidden_size=256,
        activation_fn="tanh",
        eval_freq=100_000,
        eval_episodes=5,
        max_steps=1_000,
        save_prefix="sb3_ppo_humanoid",
    )
    run_ppo_task(parser)


if __name__ == "__main__":
    main()
