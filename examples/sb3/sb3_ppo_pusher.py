"""Train and run PPO on MuJoCo Pusher-v5 using Stable-Baselines3.

Install dependency:
    pip install -r requirements.txt

Train:
    python examples/sb3/sb3_ppo_pusher.py --device cpu

Train with TensorBoard logging:
    python examples/sb3/sb3_ppo_pusher.py --device cpu --tensorboard-log logs/tensorboard

Start TensorBoard:
    tensorboard --logdir logs/tensorboard

Evaluate a saved model:
    python examples/sb3/sb3_ppo_pusher.py --no-train --load-path models/sb3_ppo_pusher.zip

Watch a saved model:
    python examples/sb3/sb3_ppo_pusher.py --no-train --load-path models/sb3_ppo_pusher.zip --watch
"""

from __future__ import annotations

import argparse

from sb3_ppo_mujoco_common import add_common_args, run_ppo_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable-Baselines3 PPO for MuJoCo Pusher-v5.")
    add_common_args(
        parser,
        env_id="Pusher-v5",
        total_timesteps=1_000_000,
        n_envs=8,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
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
        eval_freq=25_000,
        eval_episodes=5,
        max_steps=100,
        save_prefix="sb3_ppo_pusher",
    )
    run_ppo_task(parser)


if __name__ == "__main__":
    main()
