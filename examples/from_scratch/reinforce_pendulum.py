"""用纯 REINFORCE 学习 Pendulum-v1；Python 3.10+，无 Critic/GAE。

训练：python examples/from_scratch/reinforce_pendulum.py
评估：python examples/from_scratch/reinforce_pendulum.py --mode eval --eval-episodes 20
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # 必须在导入 pyplot 前设置，支持无 GUI 服务器。
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


@dataclass
class Config:
    """所有实验超参数的默认值集中在这里，命令行可以覆盖。"""
    env_id: str = "Pendulum-v1"
    episodes: int = 5000
    batch_episodes: int = 20
    gamma: float = 0.99
    lr: float = 3e-4
    hidden_size: int = 64
    log_std_init: float = -0.5
    log_std_min: float = -2.0
    log_std_max: float = 1.0
    normalize_returns: bool = False
    seed: int = 42
    device: str = "auto"
    cpu_threads: int = 1
    log_every: int = 100
    moving_window: int = 100
    eval_episodes: int = 20
    output_dir: str = "logs/reinforce_pendulum"


def set_seed(seed: int) -> None:
    """固定随机源；不同硬件/库版本之间仍不保证逐位一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PolicyNetwork(nn.Module):
    """两层 MLP 输出高斯均值；每个动作维度有独立可学习的 log_std。"""

    def __init__(self, obs_dim: int, action_low: np.ndarray,
                 action_high: np.ndarray, cfg: Config) -> None:
        super().__init__()
        action_dim = len(action_low)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, cfg.hidden_size), nn.Tanh(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.Tanh(),
            nn.Linear(cfg.hidden_size, action_dim),
        )
        # 初始均值接近零，减少初始 tanh 饱和；这不是预训练。
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), cfg.log_std_init))
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2)
        self.register_buffer("action_bias", (high + low) / 2)

    def forward(self, state: torch.Tensor) -> Normal:
        """state [3] 或 [N,3]；mu/std 为 [1] 或 [N,1]。"""
        mu = self.net(state)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        return Normal(mu, std)


def select_action(policy: PolicyNetwork, observation: np.ndarray,
                  deterministic: bool = False) -> tuple[np.ndarray, torch.Tensor | None]:
    """执行动作 [1]；训练 log_prob 为标量 []，必须保留其计算图。"""
    device = policy.action_scale.device
    state = torch.as_tensor(observation, dtype=torch.float32, device=device)
    distribution = policy(state)
    # sample() 的 u 不含参数梯度，保证采用 score-function 而非路径梯度。
    u = distribution.mean if deterministic else distribution.sample()
    action = policy.action_bias + policy.action_scale * torch.tanh(u)
    log_prob = None
    if not deterministic:
        # log(1 - tanh(u)^2) 的稳定等价形式；避免 log(0) 和额外 epsilon 偏差。
        log_tanh_jacobian = 2 * (math.log(2.0) - u - F.softplus(-2 * u))
        # 连动作缩放的 Jacobian 也计入；多维独立高斯的联合 log 密度为和。
        log_prob = (
            distribution.log_prob(u)
            - log_tanh_jacobian
            - policy.action_scale.log()
        ).sum(dim=-1)
    # 只把环境动作 detach；上面的 log_prob 不能 detach 或转换成 float。
    return action.detach().cpu().numpy().astype(np.float32), log_prob


def compute_returns(rewards: list[float], gamma: float,
                    device: torch.device) -> torch.Tensor:
    """逆序递推 G_t = r_t + gamma * G_(t+1)，返回 [T]，不建立计算图。"""
    returns = [0.0] * len(rewards)
    running_return = 0.0  # 本实验优化截断前的有限时域回报，不 bootstrap。
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return torch.tensor(returns, dtype=torch.float32, device=device)


def save_rewards(rewards: list[float], cfg: Config) -> None:
    """保存全部 Episode 原始奖励及过去 window 回合的滑动平均图。"""
    output = Path(cfg.output_dir)
    values = np.asarray(rewards, dtype=np.float64)
    x = np.arange(1, len(values) + 1)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    starts = np.maximum(0, np.arange(len(values)) - cfg.moving_window + 1)
    moving = (cumulative[1:] - cumulative[starts]) / (x - starts)
    np.savetxt(output / "episode_rewards.csv", np.column_stack((x, values)),
               delimiter=",", header="episode,reward", comments="", fmt=["%d", "%.6f"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, values, alpha=0.3, linewidth=0.8, label="Episode reward")
    ax.plot(x, moving, linewidth=2, label=f"Moving average ({cfg.moving_window})")
    ax.set(xlabel="Episode", ylabel="Undiscounted episode reward",
           title="REINFORCE on Pendulum-v1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "reward_curve.png", dpi=160)
    plt.close(fig)


def train(env: gym.Env, policy: PolicyNetwork, cfg: Config) -> list[float]:
    """同一策略采集多个完整 Episode，使用一次 Monte Carlo 梯度更新。"""
    policy.train()
    device = policy.action_scale.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    episode_rewards: list[float] = []
    next_log = cfg.log_every
    update = 0
    while len(episode_rewards) < cfg.episodes:
        batch_log_probs: list[torch.Tensor] = []
        batch_returns: list[torch.Tensor] = []
        count = min(cfg.batch_episodes, cfg.episodes - len(episode_rewards))
        for _ in range(count):
            episode_index = len(episode_rewards)
            observation, _info = env.reset(seed=cfg.seed + episode_index)
            rewards: list[float] = []
            while True:
                action, log_prob = select_action(policy, observation)
                observation, reward, terminated, truncated, _info = env.step(action)
                assert log_prob is not None
                batch_log_probs.append(log_prob)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            batch_returns.append(compute_returns(rewards, cfg.gamma, device))
            episode_rewards.append(sum(rewards))  # 日志用未折扣奖励。

        log_probs = torch.stack(batch_log_probs)  # [N]，N 为这一批的总步数。
        returns = torch.cat(batch_returns)       # [N]，与 log_probs 逐步对齐。
        if cfg.normalize_returns:
            # 可选批内标准化；不是原始 REINFORCE，也不是学习型 Baseline。
            returns = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-8)
        # 对 Episode 求平均，Episode 内按时间求和；不是除以总步数 N。
        loss = -(log_probs * returns).sum() / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite policy loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # 投影保证参数自身也在区间内，避免停留在 clamp 外侧的零梯度区。
            policy.log_std.clamp_(cfg.log_std_min, cfg.log_std_max)
        update += 1
        completed = len(episode_rewards)
        if completed >= next_log or completed == cfg.episodes:
            recent = episode_rewards[-cfg.moving_window:]
            print(f"Episode {completed:5d}/{cfg.episodes} | update {update:4d} | "
                  f"mean reward ({len(recent)}): {np.mean(recent):8.2f} | "
                  f"loss: {loss.item():.2f}", flush=True)
            while next_log <= completed:
                next_log += cfg.log_every
        # 当前 batch 的计算图用完即释放，不重复更新旧轨迹。
        del loss, log_probs, returns, log_prob, batch_log_probs, batch_returns

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state_dict": policy.state_dict(), "config": asdict(cfg)},
               output / "reinforce_pendulum.pth")
    save_rewards(episode_rewards, cfg)
    print(f"Saved model, curve and rewards to {output.resolve()}")
    return episode_rewards


@torch.no_grad()
def evaluate(env: gym.Env, policy: PolicyNetwork, episodes: int,
             seed: int) -> list[float]:
    """使用 scale * tanh(mu) + bias 确定性评估，不构建计算图。"""
    policy.eval()
    scores: list[float] = []
    for episode in range(episodes):
        observation, _info = env.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            action, _ = select_action(policy, observation, deterministic=True)
            observation, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        scores.append(total_reward)
        print(f"Eval episode {episode + 1:3d}: {total_reward:8.2f}")
    print(f"Evaluation: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}, "
          f"min={np.min(scores):.2f}, max={np.max(scores):.2f}")
    return scores


def main() -> None:
    """解析参数、设置随机种子/设备，创建环境并执行训练或加载评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true", help="仅评估时显示窗口")
    # 从 Config 自动创建参数，例如 batch_episodes 对应 --batch-episodes。
    for name, default in asdict(Config()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = parser.parse_args()
    cfg = Config(**{name: getattr(args, name) for name in asdict(Config())})
    if min(cfg.episodes, cfg.batch_episodes, cfg.eval_episodes, cfg.hidden_size,
           cfg.cpu_threads, cfg.log_every, cfg.moving_window) <= 0:
        parser.error("Episode counts, sizes, threads and logging intervals must be positive")
    if not 0 <= cfg.gamma <= 1 or cfg.lr <= 0:
        parser.error("Require gamma in [0,1] and lr > 0")
    if not cfg.log_std_min <= cfg.log_std_init <= cfg.log_std_max:
        parser.error("Require log_std_min <= log_std_init <= log_std_max")
    if cfg.env_id != "Pendulum-v1":
        parser.error("This teaching example is restricted to Pendulum-v1")
    set_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    print(f"Device: {device}; normalize_returns={cfg.normalize_returns}")
    # 在 make 时指定渲染方式；训练环境始终不打开 GUI。
    env = gym.make(cfg.env_id, render_mode="human" if args.mode == "eval" and args.render else None)
    try:
        policy_cfg = cfg
        checkpoint = None
        if args.mode == "eval":
            model_path = args.model_path or Path(cfg.output_dir) / "reinforce_pendulum.pth"
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            policy_cfg = Config(**checkpoint["config"])  # 恢复网络宽度等元数据。
            if policy_cfg.env_id != cfg.env_id:
                raise ValueError("Checkpoint environment does not match")
        policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.low,
                               env.action_space.high, policy_cfg).to(device)
        if checkpoint is not None:
            policy.load_state_dict(checkpoint["policy_state_dict"])
        if args.mode == "train":
            train(env, policy, cfg)
        # 评估随机种子与训练所用种子区间分开；不同模型比较应固定该种子。
        evaluate(env, policy, cfg.eval_episodes, cfg.seed + 1_000_000)
    finally:
        env.close()


if __name__ == "__main__":
    main()
