"""PPO solution for the discrete LunarLander-v3 environment.

Install dependency:
    pip install -r requirements.txt

Run training:
    python examples/from_scratch/ppo_lunarlander.py

Watch after training:
    python examples/from_scratch/ppo_lunarlander.py --watch

Evaluate a saved model:
    python examples/from_scratch/ppo_lunarlander.py --load-path models/ppo_lunarlander.pt --no-train --watch
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
    from torch.distributions import Categorical
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


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(states)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

    def get_action_and_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self(states)
        distribution = Categorical(logits=logits)

        if actions is None:
            actions = distribution.sample()

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return actions, log_probs, entropy, values


def make_env(env_id: str, render_mode: str | None = None) -> Any:
    try:
        return gym.make(env_id, render_mode=render_mode)
    except Exception as exc:
        if "Box2D" in str(exc) or "box2d" in str(exc).lower() or "swig" in str(exc).lower():
            raise SystemExit(
                "LunarLander needs Box2D dependencies. Install them with:\n"
                "    pip install -r requirements.txt\n"
                "If installation fails on Windows, try:\n"
                "    pip install swig\n"
                "    pip install gymnasium[box2d]"
            ) from exc
        raise


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


def select_action(
    model: ActorCritic,
    state: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> int:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _value = model(state_tensor)

    if deterministic:
        return int(torch.argmax(logits, dim=1).item())

    distribution = Categorical(logits=logits)
    return int(distribution.sample().item())


def collect_rollout(
    env: Any,
    model: ActorCritic,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    state: np.ndarray,
    episode_rewards: deque[float],
    seed_offset: int,
) -> tuple[RolloutBatch, np.ndarray, int, list[float]]:
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    finished_rewards = []
    current_episode_reward = episode_rewards[-1] if episode_rewards else 0.0
    resets = 0

    for _ in range(rollout_steps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob_tensor, _entropy, value_tensor = model.get_action_and_value(state_tensor)

        action = int(action_tensor.item())
        next_state, reward, done = step_env(env, action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(float(log_prob_tensor.item()))
        values.append(float(value_tensor.item()))

        current_episode_reward += reward
        state = next_state

        if done:
            finished_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            resets += 1
            state = reset_env(env, seed=seed_offset + resets)

    with torch.no_grad():
        last_state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        _logits, last_value_tensor = model(last_state_tensor)
        last_value = float(last_value_tensor.item())

    advantages = np.zeros(rollout_steps, dtype=np.float32)
    returns = np.zeros(rollout_steps, dtype=np.float32)
    gae = 0.0

    for step in reversed(range(rollout_steps)):
        next_non_terminal = 1.0 - float(dones[step])
        next_value = last_value if step == rollout_steps - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae
        returns[step] = advantages[step] + values[step]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = RolloutBatch(
        states=torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
        actions=torch.as_tensor(actions, dtype=torch.int64, device=device),
        old_log_probs=torch.as_tensor(log_probs, dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
    )
    episode_rewards.append(current_episode_reward)
    return batch, state, resets, finished_rewards


def update_ppo(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    update_epochs: int,
    minibatch_size: int,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> tuple[float, float, float]:
    batch_size = batch.states.shape[0]
    policy_losses = []
    value_losses = []
    entropy_losses = []

    for _epoch in range(update_epochs):
        indices = torch.randperm(batch_size, device=batch.states.device)

        for start in range(0, batch_size, minibatch_size):
            mb_indices = indices[start : start + minibatch_size]

            _actions, new_log_probs, entropy, values = model.get_action_and_value(
                batch.states[mb_indices],
                batch.actions[mb_indices],
            )
            log_ratio = new_log_probs - batch.old_log_probs[mb_indices]
            ratio = log_ratio.exp()

            unclipped_policy_loss = -batch.advantages[mb_indices] * ratio
            clipped_policy_loss = -batch.advantages[mb_indices] * torch.clamp(
                ratio,
                1.0 - clip_coef,
                1.0 + clip_coef,
            )
            policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()
            value_loss = nn.functional.mse_loss(values, batch.returns[mb_indices])
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_losses.append(float(entropy_loss.item()))

    return (
        float(np.mean(policy_losses)),
        float(np.mean(value_losses)),
        float(np.mean(entropy_losses)),
    )


def train_ppo(
    env: Any,
    env_id: str,
    model: ActorCritic,
    total_timesteps: int,
    rollout_steps: int,
    max_steps: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    update_epochs: int,
    minibatch_size: int,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    solved_score: float,
    eval_interval_updates: int,
    train_eval_episodes: int,
    save_path: str,
    state_dim: int,
    action_dim: int,
    seed: int,
    device: torch.device,
) -> ActorCritic:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    state = reset_env(env, seed=seed)
    recent_finished_rewards: deque[float] = deque(maxlen=100)
    ongoing_rewards: deque[float] = deque(maxlen=1)
    steps_done = 0
    update = 0
    seed_offset = seed + 10_000
    best_eval_reward = -float("inf")

    while steps_done < total_timesteps:
        batch, state, resets, finished_rewards = collect_rollout(
            env=env,
            model=model,
            rollout_steps=rollout_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
            state=state,
            episode_rewards=ongoing_rewards,
            seed_offset=seed_offset + update * 100,
        )
        steps_done += rollout_steps
        update += 1

        recent_finished_rewards.extend(finished_rewards)
        policy_loss, value_loss, entropy = update_ppo(
            model=model,
            optimizer=optimizer,
            batch=batch,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            clip_coef=clip_coef,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )

        if update == 1 or update % 5 == 0:
            avg100 = np.mean(recent_finished_rewards) if recent_finished_rewards else float("nan")
            last_reward = recent_finished_rewards[-1] if recent_finished_rewards else float("nan")
            print(
                f"Update {update:4d} | "
                f"steps {steps_done:7d} | "
                f"episodes {len(recent_finished_rewards):3d} | "
                f"last {last_reward:8.1f} | "
                f"avg100 {avg100:8.1f} | "
                f"policy {policy_loss:8.4f} | "
                f"value {value_loss:8.2f} | "
                f"entropy {entropy:6.3f} | "
                f"resets {resets:2d}"
            )

        if update == 1 or update % eval_interval_updates == 0:
            deterministic_eval = evaluate_policy(
                env_id=env_id,
                model=model,
                episodes=train_eval_episodes,
                max_steps=max_steps,
                seed=seed + 60_000 + update * 100,
                device=device,
                deterministic=True,
            )
            stochastic_eval = evaluate_policy(
                env_id=env_id,
                model=model,
                episodes=train_eval_episodes,
                max_steps=max_steps,
                seed=seed + 70_000 + update * 100,
                device=device,
                deterministic=False,
            )
            print(
                f"Eval   {update:4d} | "
                f"deterministic {deterministic_eval:8.1f} | "
                f"stochastic {stochastic_eval:8.1f}"
            )

            if deterministic_eval > best_eval_reward:
                best_eval_reward = deterministic_eval
                save_model(model, save_path, env_id, state_dim, action_dim)
                print(f"Saved best model to: {save_path} | eval {best_eval_reward:.1f}")

            if deterministic_eval >= solved_score:
                print(f"Solved at {steps_done} steps. Deterministic eval reward: {deterministic_eval:.1f}")
                break

    return model


def evaluate_policy(
    env_id: str,
    model: ActorCritic,
    episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
    deterministic: bool = True,
) -> float:
    env = make_env(env_id)
    rewards = []

    for episode in range(episodes):
        state = reset_env(env, seed=seed + episode)
        episode_reward = 0.0

        for _step in range(max_steps):
            action = select_action(model, state, device, deterministic)
            state, reward, done = step_env(env, action)
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

    env.close()
    return float(np.mean(rewards))


def watch_policy(
    env_id: str,
    model: ActorCritic,
    episodes: int,
    max_steps: int,
    delay: float,
    seed: int,
    device: torch.device,
    deterministic: bool,
) -> None:
    env = make_env(env_id, render_mode="human")

    for episode in range(episodes):
        state = reset_env(env, seed=seed + episode)

        for _step in range(max_steps):
            action = select_action(model, state, device, deterministic)
            state, _reward, done = step_env(env, action)
            env.render()
            time.sleep(delay)

            if done:
                break

        time.sleep(1.0)

    env.close()


def save_model(model: ActorCritic, path: str, env_id: str, state_dim: int, action_dim: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "env_id": env_id,
            "state_dim": state_dim,
            "action_dim": action_dim,
        },
        path,
    )


def load_model(path: str, device: torch.device) -> tuple[ActorCritic, str]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise SystemExit(f"Model file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dim = int(checkpoint.get("state_dim", 8))
    action_dim = int(checkpoint.get("action_dim", 4))
    env_id = str(checkpoint.get("env_id", "LunarLander-v3"))

    model = ActorCritic(state_dim, action_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, env_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3.")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--rollout-steps", type=int, default=2_048)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--solved-score", type=float, default=200.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--train-eval-episodes", type=int, default=10)
    parser.add_argument("--eval-interval-updates", type=int, default=10)
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="models/ppo_lunarlander.pt")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-episodes", type=int, default=1)
    parser.add_argument("--watch-delay", type=float, default=0.02)
    parser.add_argument("--stochastic-watch", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.load_path:
        model, loaded_env_id = load_model(args.load_path, device)
        args.env_id = loaded_env_id
        print(f"Loaded model from: {args.load_path}")
    else:
        env = make_env(args.env_id)
        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)
        env.close()
        model = ActorCritic(state_dim, action_dim).to(device)

    if not args.no_train:
        env = make_env(args.env_id)
        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)

        model = train_ppo(
            env=env,
            env_id=args.env_id,
            model=model,
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            clip_coef=args.clip_coef,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            solved_score=args.solved_score,
            eval_interval_updates=args.eval_interval_updates,
            train_eval_episodes=args.train_eval_episodes,
            save_path=args.save_path,
            state_dim=state_dim,
            action_dim=action_dim,
            seed=args.seed,
            device=device,
        )
        env.close()
        model, _loaded_env_id = load_model(args.save_path, device)
        print(f"Loaded best model from: {args.save_path}")

    avg_reward = evaluate_policy(
        env_id=args.env_id,
        model=model,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed + 50_000,
        device=device,
        deterministic=not args.stochastic_eval,
    )
    eval_mode = "stochastic" if args.stochastic_eval else "deterministic"
    print(f"\nEvaluation average reward ({eval_mode}): {avg_reward:.1f}")

    if args.watch:
        watch_policy(
            env_id=args.env_id,
            model=model,
            episodes=args.watch_episodes,
            max_steps=args.max_steps,
            delay=args.watch_delay,
            seed=args.seed + 80_000,
            device=device,
            deterministic=not args.stochastic_watch,
        )


if __name__ == "__main__":
    main()
