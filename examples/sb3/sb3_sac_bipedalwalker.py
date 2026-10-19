"""Train and run SAC on BipedalWalker-v3 using Stable-Baselines3.

Install dependency:
    pip install -r requirements.txt

Train:
    python examples/sb3/sb3_sac_bipedalwalker.py --device cpu

Train with TensorBoard logging:
    python examples/sb3/sb3_sac_bipedalwalker.py --device cpu --tensorboard-log logs/tensorboard

Start TensorBoard:
    tensorboard --logdir logs/tensorboard

Evaluate a saved model:
    python examples/sb3/sb3_sac_bipedalwalker.py --no-train --load-path models/sb3_sac_bipedalwalker.zip

Watch a saved model:
    python examples/sb3/sb3_sac_bipedalwalker.py --no-train --load-path models/sb3_sac_bipedalwalker.zip --watch
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    import torch as th
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with:\n"
        "    pip install -r requirements.txt"
    ) from exc


def make_env(env_id: str, seed: int, render_mode: str | None = None) -> Any:
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except Exception as exc:
        if "box2d" in str(exc).lower() or "swig" in str(exc).lower():
            raise SystemExit(
                "BipedalWalker needs Box2D dependencies. Install them with:\n"
                "    pip install -r requirements.txt"
            ) from exc
        raise

    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def train_model(args: argparse.Namespace) -> SAC:
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    train_env = make_env(args.env_id, args.seed)
    eval_env = make_env(args.env_id, args.seed + 10_000)

    callback = EvalCallback(
        eval_env,
        best_model_save_path=args.best_model_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        use_sde=args.use_sde,
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": [args.hidden_size_1, args.hidden_size_2],
            "log_std_init": args.log_std_init,
        },
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=args.log_interval,
        progress_bar=args.progress_bar,
    )
    model.save(args.save_path)
    print(f"Saved final model to: {args.save_path}")

    train_env.close()
    eval_env.close()

    best_model_path = Path(args.best_model_dir) / "best_model.zip"
    if best_model_path.exists():
        print(f"Loading best evaluated model from: {best_model_path}")
        return SAC.load(best_model_path, device=args.device)

    return model


def load_model(path: str, device: str) -> SAC:
    model_path = Path(path)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    return SAC.load(model_path, device=device)


def evaluate_model(model: SAC, args: argparse.Namespace) -> None:
    env = make_env(args.env_id, args.seed + 20_000)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.eval_episodes,
        deterministic=not args.stochastic_eval,
    )
    env.close()

    mode = "stochastic" if args.stochastic_eval else "deterministic"
    print(f"Evaluation mode: {mode}")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def watch_model(model: SAC, args: argparse.Namespace) -> None:
    env = make_env(args.env_id, args.seed + 30_000, render_mode="human")

    for episode in range(args.watch_episodes):
        obs, _info = env.reset(seed=args.seed + 30_000 + episode)
        episode_reward = 0.0

        for _step in range(args.max_steps):
            action, _state = model.predict(obs, deterministic=not args.stochastic_watch)
            obs, reward, terminated, truncated, _info = env.step(action)
            episode_reward += float(reward)
            env.render()
            time.sleep(args.watch_delay)

            if terminated or truncated:
                break

        print(f"Watch episode {episode + 1} reward: {episode_reward:.2f}")
        time.sleep(1.0)

    env.close()


def infer_single_action(model: SAC, state_text: str, deterministic: bool) -> None:
    values = [float(item.strip()) for item in state_text.split(",")]
    if len(values) != 24:
        raise SystemExit("BipedalWalker-v3 state must contain exactly 24 numbers.")

    obs = np.asarray(values, dtype=np.float32)
    action, _state = model.predict(obs, deterministic=deterministic)
    print(f"State: {obs.tolist()}")
    print(f"Action: {np.asarray(action).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable-Baselines3 SAC for BipedalWalker-v3.")
    parser.add_argument("--env-id", type=str, default="BipedalWalker-v3")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=7.3e-4)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--train-freq", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=64)
    parser.add_argument("--ent-coef", type=str, default="auto")
    parser.add_argument("--use-sde", action="store_true", default=True)
    parser.add_argument("--no-use-sde", action="store_false", dest="use_sde")
    parser.add_argument("--log-std-init", type=float, default=-3.0)
    parser.add_argument("--hidden-size-1", type=int, default=400)
    parser.add_argument("--hidden-size-2", type=int, default=300)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1_600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default="models/sb3_sac_bipedalwalker.zip")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--best-model-dir", type=str, default="logs/sb3_sac_bipedalwalker/best")
    parser.add_argument("--log-dir", type=str, default="logs/sb3_sac_bipedalwalker")
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=4)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-episodes", type=int, default=1)
    parser.add_argument("--watch-delay", type=float, default=0.02)
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--stochastic-watch", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--state", type=str, default=None)
    args = parser.parse_args()

    if args.load_path:
        model = load_model(args.load_path, args.device)
        print(f"Loaded model from: {args.load_path}")
    elif args.no_train:
        model = load_model(args.save_path, args.device)
        print(f"Loaded model from: {args.save_path}")
    else:
        model = train_model(args)

    if args.state is not None:
        infer_single_action(model, args.state, deterministic=not args.stochastic_eval)
        return

    evaluate_model(model, args)

    if args.watch:
        watch_model(model, args)


if __name__ == "__main__":
    main()
