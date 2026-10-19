"""Train and run SAC on BipedalWalker-v3 without Stable-Baselines3.

Install dependency:
    pip install -r requirements.txt

Train:
    python examples/from_scratch/sac_bipedalwalker.py --device cpu

Train with TensorBoard logging:
    python examples/from_scratch/sac_bipedalwalker.py --device cpu --tensorboard-log logs/tensorboard/manual_sac_bipedalwalker

Start TensorBoard:
    tensorboard --logdir logs/tensorboard

Evaluate a saved model:
    python examples/from_scratch/sac_bipedalwalker.py --no-train --load-path models/sac_bipedalwalker.pt

Watch a saved model:
    python examples/from_scratch/sac_bipedalwalker.py --no-train --load-path models/sac_bipedalwalker.pt --watch
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
    from torch.distributions import Normal
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with:\n"
        "    pip install -r requirements.txt"
    ) from exc

try:
    import gymnasium as gym
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with:\n"
        "    pip install -r requirements.txt"
    ) from exc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPSILON = 1e-6


@dataclass
class SacModels:
    actor: "GaussianActor"
    q1: "QNetwork"
    q2: "QNetwork"
    q1_target: "QNetwork"
    q2_target: "QNetwork"
    log_alpha: torch.Tensor


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return self.size


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(states)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        tanh_z = torch.tanh(z)
        action = tanh_z * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1.0 - tanh_z.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=1))


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

    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def reset_env(env: Any, seed: int | None = None) -> np.ndarray:
    observation, _info = env.reset(seed=seed)
    return np.asarray(observation, dtype=np.float32)


def step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
    observation, reward, terminated, truncated, _info = env.step(action)
    done = terminated or truncated
    return np.asarray(observation, dtype=np.float32), float(reward), bool(done)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def build_models(
    state_dim: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    device: torch.device,
) -> SacModels:
    actor = GaussianActor(state_dim, action_dim, action_low, action_high).to(device)
    q1 = QNetwork(state_dim, action_dim).to(device)
    q2 = QNetwork(state_dim, action_dim).to(device)
    q1_target = QNetwork(state_dim, action_dim).to(device)
    q2_target = QNetwork(state_dim, action_dim).to(device)
    hard_update(q1_target, q1)
    hard_update(q2_target, q2)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    return SacModels(actor, q1, q2, q1_target, q2_target, log_alpha)


def select_action(
    actor: GaussianActor,
    state: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        sampled_action, _log_prob, deterministic_action = actor.sample(state_tensor)
    action = deterministic_action if deterministic else sampled_action
    return action.squeeze(0).cpu().numpy()


def update_sac(
    models: SacModels,
    replay_buffer: ReplayBuffer,
    actor_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    tau: float,
    target_entropy: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device)
    alpha = models.log_alpha.exp()

    with torch.no_grad():
        next_actions, next_log_probs, _next_mean_actions = models.actor.sample(next_states)
        target_q1 = models.q1_target(next_states, next_actions)
        target_q2 = models.q2_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
        q_target = rewards + gamma * (1.0 - dones) * target_q

    q1_values = models.q1(states, actions)
    q2_values = models.q2(states, actions)
    q1_loss = nn.functional.mse_loss(q1_values, q_target)
    q2_loss = nn.functional.mse_loss(q2_values, q_target)
    q_loss = q1_loss + q2_loss

    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    new_actions, log_probs, _mean_actions = models.actor.sample(states)
    min_q_new_actions = torch.min(models.q1(states, new_actions), models.q2(states, new_actions))
    actor_loss = (alpha.detach() * log_probs - min_q_new_actions).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    alpha_loss = -(models.log_alpha * (log_probs + target_entropy).detach()).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    soft_update(models.q1_target, models.q1, tau)
    soft_update(models.q2_target, models.q2, tau)

    return (
        float(q_loss.item()),
        float(actor_loss.item()),
        float(alpha_loss.item()),
        float(models.log_alpha.exp().item()),
    )


def evaluate_policy(
    env_id: str,
    actor: GaussianActor,
    episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
    deterministic: bool,
) -> float:
    env = make_env(env_id, seed)
    rewards = []

    for episode in range(episodes):
        state = reset_env(env, seed=seed + episode)
        episode_reward = 0.0

        for _step in range(max_steps):
            action = select_action(actor, state, device, deterministic)
            state, reward, done = step_env(env, action)
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)

    env.close()
    return float(np.mean(rewards))


def save_checkpoint(
    path: str,
    models: SacModels,
    env_id: str,
    state_dim: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": models.actor.state_dict(),
            "q1": models.q1.state_dict(),
            "q2": models.q2.state_dict(),
            "q1_target": models.q1_target.state_dict(),
            "q2_target": models.q2_target.state_dict(),
            "log_alpha": models.log_alpha.detach().cpu(),
            "env_id": env_id,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[SacModels, str]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise SystemExit(f"Model file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    models = build_models(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        action_low=np.asarray(checkpoint["action_low"], dtype=np.float32),
        action_high=np.asarray(checkpoint["action_high"], dtype=np.float32),
        device=device,
    )
    models.actor.load_state_dict(checkpoint["actor"])
    models.q1.load_state_dict(checkpoint["q1"])
    models.q2.load_state_dict(checkpoint["q2"])
    models.q1_target.load_state_dict(checkpoint["q1_target"])
    models.q2_target.load_state_dict(checkpoint["q2_target"])
    models.log_alpha.data.copy_(checkpoint["log_alpha"].to(device))

    models.actor.eval()
    models.q1.eval()
    models.q2.eval()
    models.q1_target.eval()
    models.q2_target.eval()
    return models, str(checkpoint["env_id"])


def train(args: argparse.Namespace, device: torch.device) -> SacModels:
    env = make_env(args.env_id, args.seed)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)
    models = build_models(state_dim, action_dim, action_low, action_high, device)

    actor_optimizer = torch.optim.Adam(models.actor.parameters(), lr=args.learning_rate)
    q_optimizer = torch.optim.Adam(list(models.q1.parameters()) + list(models.q2.parameters()), lr=args.learning_rate)
    alpha_optimizer = torch.optim.Adam([models.log_alpha], lr=args.learning_rate)
    target_entropy = -float(action_dim) if args.target_entropy == "auto" else float(args.target_entropy)
    replay_buffer = ReplayBuffer(state_dim, action_dim, args.buffer_size)
    writer = None
    if args.tensorboard_log:
        if SummaryWriter is None:
            raise SystemExit("TensorBoard is not installed. Run: pip install -r requirements.txt")
        writer = SummaryWriter(args.tensorboard_log)

    state = reset_env(env, seed=args.seed)
    episode_reward = 0.0
    episode = 1
    recent_rewards: deque[float] = deque(maxlen=20)
    best_eval_reward = -float("inf")
    last_losses = (float("nan"), float("nan"), float("nan"), float("nan"))

    for step in range(1, args.total_timesteps + 1):
        if step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            action = select_action(models.actor, state, device, deterministic=False)

        next_state, reward, done = step_env(env, action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            if episode == 1 or episode % args.log_interval == 0:
                avg20 = float(np.mean(recent_rewards))
                print(
                    f"Episode {episode:5d} | "
                    f"step {step:8d} | "
                    f"reward {episode_reward:8.1f} | "
                    f"avg20 {avg20:8.1f} | "
                    f"q {last_losses[0]:8.2f} | "
                    f"actor {last_losses[1]:8.2f} | "
                    f"alpha {last_losses[3]:6.3f}"
                )
            if writer:
                writer.add_scalar("rollout/episode_reward", episode_reward, step)
                writer.add_scalar("rollout/avg20_reward", float(np.mean(recent_rewards)), step)

            state = reset_env(env, seed=args.seed + episode)
            episode_reward = 0.0
            episode += 1

        if step > args.learning_starts and len(replay_buffer) >= args.batch_size and step % args.update_every == 0:
            for _ in range(args.gradient_steps):
                last_losses = update_sac(
                    models=models,
                    replay_buffer=replay_buffer,
                    actor_optimizer=actor_optimizer,
                    q_optimizer=q_optimizer,
                    alpha_optimizer=alpha_optimizer,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    tau=args.tau,
                    target_entropy=target_entropy,
                    device=device,
                )
            if writer:
                q_loss, actor_loss, alpha_loss, alpha = last_losses
                writer.add_scalar("train/q_loss", q_loss, step)
                writer.add_scalar("train/actor_loss", actor_loss, step)
                writer.add_scalar("train/alpha_loss", alpha_loss, step)
                writer.add_scalar("train/alpha", alpha, step)

        if step == args.learning_starts or step % args.eval_freq == 0:
            eval_reward = evaluate_policy(
                args.env_id,
                models.actor,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
                seed=args.seed + 20_000 + step,
                device=device,
                deterministic=True,
            )
            print(f"Eval step {step:8d} | deterministic reward {eval_reward:8.1f}")
            if writer:
                writer.add_scalar("eval/mean_reward", eval_reward, step)

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_checkpoint(args.save_path, models, args.env_id, state_dim, action_dim, action_low, action_high)
                print(f"Saved best model to: {args.save_path} | eval {best_eval_reward:.1f}")

            if eval_reward >= args.solved_score:
                print(f"Solved at step {step}. Evaluation reward: {eval_reward:.1f}")
                break

    env.close()
    if writer:
        writer.close()

    best_path = Path(args.save_path)
    if best_path.exists():
        models, _env_id = load_checkpoint(args.save_path, device)
    else:
        save_checkpoint(args.save_path, models, args.env_id, state_dim, action_dim, action_low, action_high)
    return models


def watch_policy(
    env_id: str,
    actor: GaussianActor,
    episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
    delay: float,
    deterministic: bool,
) -> None:
    env = make_env(env_id, seed, render_mode="human")

    for episode in range(episodes):
        state = reset_env(env, seed=seed + episode)
        episode_reward = 0.0

        for _step in range(max_steps):
            action = select_action(actor, state, device, deterministic)
            state, reward, done = step_env(env, action)
            episode_reward += reward
            env.render()
            time.sleep(delay)
            if done:
                break

        print(f"Watch episode {episode + 1} reward: {episode_reward:.1f}")
        time.sleep(1.0)

    env.close()


def infer_single_action(actor: GaussianActor, state_text: str, device: torch.device, deterministic: bool) -> None:
    values = [float(item.strip()) for item in state_text.split(",")]
    if len(values) != 24:
        raise SystemExit("BipedalWalker-v3 state must contain exactly 24 numbers.")
    state = np.asarray(values, dtype=np.float32)
    action = select_action(actor, state, device, deterministic)
    print(f"State: {state.tolist()}")
    print(f"Action: {action.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual PyTorch SAC for BipedalWalker-v3.")
    parser.add_argument("--env-id", type=str, default="BipedalWalker-v3")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-entropy", type=str, default="auto")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1_600)
    parser.add_argument("--solved-score", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default="models/sac_bipedalwalker.pt")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-episodes", type=int, default=1)
    parser.add_argument("--watch-delay", type=float, default=0.02)
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--stochastic-watch", action="store_true")
    parser.add_argument("--state", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if args.load_path:
        models, args.env_id = load_checkpoint(args.load_path, device)
        print(f"Loaded model from: {args.load_path}")
    elif args.no_train:
        models, args.env_id = load_checkpoint(args.save_path, device)
        print(f"Loaded model from: {args.save_path}")
    else:
        models = train(args, device)

    if args.state:
        infer_single_action(models.actor, args.state, device, deterministic=not args.stochastic_eval)
        return

    avg_reward = evaluate_policy(
        env_id=args.env_id,
        actor=models.actor,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed + 40_000,
        device=device,
        deterministic=not args.stochastic_eval,
    )
    mode = "stochastic" if args.stochastic_eval else "deterministic"
    print(f"\nEvaluation average reward ({mode}): {avg_reward:.1f}")

    if args.watch:
        watch_policy(
            env_id=args.env_id,
            actor=models.actor,
            episodes=args.watch_episodes,
            max_steps=args.max_steps,
            seed=args.seed + 50_000,
            device=device,
            delay=args.watch_delay,
            deterministic=not args.stochastic_watch,
        )


if __name__ == "__main__":
    main()
