"""Load a saved DQN CartPole model and run inference.

Run evaluation without rendering:
    python examples/from_scratch/infer_cartpole.py --model-path models/dqn_cartpole.pt

Watch the trained agent:
    python examples/from_scratch/infer_cartpole.py --model-path models/dqn_cartpole.pt --watch

Infer one action from a manually provided state:
    python examples/from_scratch/infer_cartpole.py --state "0.0,0.0,0.05,0.0"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with:\n"
        "    pip install -r requirements.txt"
    ) from exc

try:
    import gymnasium as gym
except ImportError:
    try:
        import gym  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from exc

from dqn_cartpole import QNetwork


def reset_env(env: Any, seed: int | None = None) -> np.ndarray:
    result = env.reset(seed=seed)
    if isinstance(result, tuple):
        observation, _info = result
    else:
        observation = result
    return np.asarray(observation, dtype=np.float32)


def step_env(env: Any, action: int) -> tuple[np.ndarray, float, bool]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, _info = result
        done = terminated or truncated
    else:
        observation, reward, done, _info = result
    return np.asarray(observation, dtype=np.float32), float(reward), bool(done)


def load_model(model_path: str, device: torch.device) -> QNetwork:
    env = gym.make("CartPole-v1")
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    env.close()

    model = QNetwork(state_dim, action_dim).to(device)

    path = Path(model_path)
    if not path.exists():
        raise SystemExit(f"Model file not found: {path}")

    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_action(model: QNetwork, state: np.ndarray, device: torch.device) -> tuple[int, np.ndarray]:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)

    action = int(torch.argmax(q_values, dim=1).item())
    return action, q_values.squeeze(0).cpu().numpy()


def parse_state(state_text: str) -> np.ndarray:
    values = [float(item.strip()) for item in state_text.split(",")]
    if len(values) != 4:
        raise SystemExit("CartPole state must contain exactly 4 numbers.")
    return np.asarray(values, dtype=np.float32)


def run_episode(
    model: QNetwork,
    env: Any,
    max_steps: int,
    device: torch.device,
    seed: int | None,
    delay: float,
) -> float:
    state = reset_env(env, seed=seed)
    episode_reward = 0.0

    for _step in range(max_steps):
        action, _q_values = select_action(model, state, device)
        state, reward, done = step_env(env, action)
        episode_reward += reward

        if delay > 0:
            env.render()
            time.sleep(delay)

        if done:
            break

    return episode_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a saved DQN CartPole model.")
    parser.add_argument("--model-path", type=str, default="models/dqn_cartpole.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--watch", action="store_true", help="Render the inference episodes.")
    parser.add_argument("--watch-delay", type=float, default=0.02)
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help='Infer one action from a state, for example: "0.0,0.0,0.05,0.0".',
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    print(f"Loaded model from: {args.model_path}")
    print(f"Using device: {device}")

    if args.state is not None:
        state = parse_state(args.state)
        action, q_values = select_action(model, state, device)
        print(f"State: {state.tolist()}")
        print(f"Q-values: {q_values.tolist()}")
        print(f"Selected action: {action}")
        return

    render_mode = "human" if args.watch else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    rewards = []

    for episode in range(args.episodes):
        reward = run_episode(
            model=model,
            env=env,
            max_steps=args.max_steps,
            device=device,
            seed=args.seed + episode,
            delay=args.watch_delay if args.watch else 0.0,
        )
        rewards.append(reward)
        print(f"Episode {episode + 1:3d} reward: {reward:.1f}")

    env.close()
    print(f"\nAverage reward: {float(np.mean(rewards)):.1f}")


if __name__ == "__main__":
    main()
