"""DQN solution for the CartPole environment.

Install dependency:
    pip install -r requirements.txt

Run:
    python examples/from_scratch/dqn_cartpole.py

Watch after training:
    python examples/from_scratch/dqn_cartpole.py --watch
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with:\n"
        "    pip install -r requirements.txt"
    ) from exc

try:
    import gymnasium as gym
except ImportError:  # Fallback for older installations.
    try:
        import gym  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from exc


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def choose_action(
    policy_net: QNetwork,
    state: np.ndarray,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def optimize_model(
    policy_net: QNetwork,
    target_net: QNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float | None:
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)

    states = torch.as_tensor(
        np.array([item.state for item in batch]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        [item.action for item in batch],
        dtype=torch.int64,
        device=device,
    ).unsqueeze(1)
    rewards = torch.as_tensor(
        [item.reward for item in batch],
        dtype=torch.float32,
        device=device,
    )
    next_states = torch.as_tensor(
        np.array([item.next_state for item in batch]),
        dtype=torch.float32,
        device=device,
    )
    dones = torch.as_tensor(
        [item.done for item in batch],
        dtype=torch.float32,
        device=device,
    )

    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1).values
        target_q_values = rewards + gamma * next_q_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def train_dqn(
    env: Any,
    episodes: int,
    max_steps: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    min_epsilon: float,
    epsilon_decay: float,
    buffer_size: int,
    batch_size: int,
    target_update_interval: int,
    seed: int,
    device: torch.device,
) -> QNetwork:
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    recent_rewards: deque[float] = deque(maxlen=20)
    total_steps = 0

    for episode in range(1, episodes + 1):
        state = reset_env(env, seed=seed + episode)
        episode_reward = 0.0
        last_loss = None

        for _step in range(max_steps):
            action = choose_action(policy_net, state, epsilon, action_dim, device)
            next_state, reward, done = step_env(env, action)

            replay_buffer.push(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )

            last_loss = optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                replay_buffer=replay_buffer,
                optimizer=optimizer,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
            )

            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        recent_rewards.append(episode_reward)

        if episode == 1 or episode % 10 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            loss_text = "n/a" if last_loss is None else f"{last_loss:.4f}"
            print(
                f"Episode {episode:4d} | "
                f"reward {episode_reward:6.1f} | "
                f"avg20 {avg_reward:6.1f} | "
                f"epsilon {epsilon:.3f} | "
                f"loss {loss_text}"
            )

        if len(recent_rewards) == recent_rewards.maxlen and np.mean(recent_rewards) >= 475:
            print(f"Solved after {episode} episodes. Average reward: {np.mean(recent_rewards):.1f}")
            break

    return policy_net


def evaluate_policy(
    env: Any,
    policy_net: QNetwork,
    episodes: int,
    max_steps: int,
    device: torch.device,
) -> float:
    rewards = []

    for episode in range(episodes):
        state = reset_env(env, seed=30_000 + episode)
        episode_reward = 0.0

        for _step in range(max_steps):
            action = choose_action(policy_net, state, epsilon=0.0, action_dim=env.action_space.n, device=device)
            state, reward, done = step_env(env, action)
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

    return float(np.mean(rewards))


def watch_policy(
    policy_net: QNetwork,
    max_steps: int,
    episodes: int,
    delay: float,
    device: torch.device,
) -> None:
    env = gym.make("CartPole-v1", render_mode="human")

    for episode in range(episodes):
        state = reset_env(env, seed=40_000 + episode)

        for _step in range(max_steps):
            action = choose_action(policy_net, state, epsilon=0.0, action_dim=env.action_space.n, device=device)
            state, _reward, done = step_env(env, action)
            env.render()
            time.sleep(delay)

            if done:
                break

        time.sleep(1.0)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent on CartPole.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="models/dqn_cartpole.pt")
    parser.add_argument("--watch", action="store_true", help="Show a render window after training.")
    parser.add_argument("--watch-episodes", type=int, default=1)
    parser.add_argument("--watch-delay", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("CartPole-v1")
    policy_net = train_dqn(
        env=env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        seed=args.seed,
        device=device,
    )

    avg_reward = evaluate_policy(
        env=env,
        policy_net=policy_net,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        device=device,
    )
    print(f"\nEvaluation average reward: {avg_reward:.1f}")

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), args.save_path)
    print(f"Saved model to: {args.save_path}")

    env.close()

    if args.watch:
        watch_policy(
            policy_net=policy_net,
            max_steps=args.max_steps,
            episodes=args.watch_episodes,
            delay=args.watch_delay,
            device=device,
        )


if __name__ == "__main__":
    main()
