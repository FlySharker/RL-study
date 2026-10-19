#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
n-step Advantage Actor-Critic (A2C) 用于 LunarLander-v3 连续动作控制
=====================================================================

这是一个从零实现的 on-policy Actor-Critic 教学项目，不依赖 Stable-Baselines3。

算法要点
--------
1. 两个独立网络：
   - Actor : s -> (mu, log_std)，即对角高斯策略的参数
   - Critic: s -> V(s)，标量状态价值
2. 连续动作：u ~ Normal(mu, std)，action = bias + scale * tanh(u)
   log_prob 包含 tanh 与动作缩放的 Jacobian 修正，并在动作维度上求和。
3. 每收集 n_steps 步数据就做**一次**更新，更新后数据立即丢弃（无经验回放）。
   n_steps=1 即退化为一步 TD 的 Actor-Critic。
4. 回报目标 R_t = r_t + gamma*r_{t+1} + ... + gamma^k * V(s_{t+k})，
   其中 k 受 rollout 末尾与 episode 边界限制。
   - terminated（真实终止）：不 bootstrap，R_t 在终止处截断
   - truncated（时间限制截断）：用截断状态的价值 bootstrap
   - 普通 rollout 截止：用 rollout 末尾后续状态的价值 bootstrap
   回报递推**绝不跨越 episode 边界**（reset 后的新状态不会被当作上一段的后续状态）。

命令
----
  python examples/from_scratch/actor_critic_lunarlander.py verify                       # 环境/形状/边界/数值自检
  python examples/from_scratch/actor_critic_lunarlander.py train --total-timesteps 20000 # 短时间试运行
  python examples/from_scratch/actor_critic_lunarlander.py train                         # 完整训练
  python examples/from_scratch/actor_critic_lunarlander.py evaluate --model-path ...     # 评估
  python examples/from_scratch/actor_critic_lunarlander.py render   --model-path ...     # 可视化 / 导出视频

作者注：代码按 "环境信息 -> 网络 -> 动作采样 -> rollout -> 回报 -> 更新 -> 评估 -> 训练循环"
的顺序分段，可以逐段阅读。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

# 让输出（含中文与重定向到文件时）统一为 UTF-8。
# 在真正的 Windows 控制台上 sys.stdout.encoding 本来就是 utf-8，这里是空操作；
# 只有在管道 / 重定向时才起作用，避免 cp936 造成乱码。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ENV_ID = "LunarLander-v3"

# log_std 的数值范围。下界防止 std 塌缩到 0（log_prob -> +inf），
# 上界防止 std 爆炸（动作完全随机、log_prob -> -inf）。
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


# =====================================================================
# 0. 环境信息：不写死任何维度或动作边界，全部从 env 对象读取
# =====================================================================
def make_env(render_mode: Optional[str] = None) -> gym.Env:
    """创建连续动作模式的 LunarLander 环境。"""
    env = gym.make(ENV_ID, continuous=True, render_mode=render_mode)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise RuntimeError(
            f"期望连续动作空间 Box，实际得到 {type(env.action_space)}。"
            "请确认使用了 continuous=True。"
        )
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise RuntimeError(f"期望 Box 观测空间，实际得到 {type(env.observation_space)}")
    return env


def env_info_from(env: gym.Env) -> Dict[str, object]:
    """把环境的输入输出维度、动作边界等抽成纯 Python 对象（便于存进模型文件）。"""
    obs_space = env.observation_space
    act_space = env.action_space
    spec = env.spec
    return {
        "env_id": spec.id if spec is not None else ENV_ID,
        "continuous": bool(getattr(env.unwrapped, "continuous", True)),
        "obs_dim": int(np.prod(obs_space.shape)),
        "obs_shape": [int(x) for x in obs_space.shape],
        "obs_low": [float(x) for x in np.asarray(obs_space.low).ravel()],
        "obs_high": [float(x) for x in np.asarray(obs_space.high).ravel()],
        "action_dim": int(np.prod(act_space.shape)),
        "action_shape": [int(x) for x in act_space.shape],
        "action_low": [float(x) for x in np.asarray(act_space.low).ravel()],
        "action_high": [float(x) for x in np.asarray(act_space.high).ravel()],
        "max_episode_steps": int(spec.max_episode_steps) if spec and spec.max_episode_steps else None,
    }


def describe_env(info: Dict[str, object]) -> str:
    """人类可读的环境说明（维度不写死，全部来自 info）。"""
    lines = []
    lines.append(f"环境: {info['env_id']}  (continuous={info['continuous']})")
    lines.append(f"观测空间: Box{tuple(info['obs_shape'])}  动作空间: Box{tuple(info['action_shape'])}")
    lines.append("  观测各维语义 (LunarLander 固定为 8 维):")
    for i, name in enumerate(
        ["x 坐标", "y 坐标", "x 线速度", "y 线速度", "倾角", "角速度", "左腿触地", "右腿触地"]
    ):
        if i < info["obs_dim"]:
            lines.append(f"    [{i}] {name:8s} 范围 [{info['obs_low'][i]:.4g}, {info['obs_high'][i]:.4g}]")
    lines.append("  动作各维语义 (continuous=True):")
    lines.append("    [0] 主引擎油门: a0 > 0 时点火, 实际功率 = (clip(a0,0,1)+1)/2 属于 [0.5, 1.0]")
    lines.append("                   a0 <= 0 时主引擎完全关闭 (注意: 不是线性映射)")
    lines.append("    [1] 侧向姿态引擎: sign(a1) 决定方向(正=左, 负=右), |a1| 被 clip 到 [0.5,1.0] 作为功率")
    lines.append("                   只要 a1 != 0 就会点火, 很小的值也会以 50% 功率喷火")
    lines.append("  动作范围: Box 上下界由 env.action_space 给出, 代码不写死")
    lines.append(f"  时间限制: max_episode_steps = {info['max_episode_steps']} (由 TimeLimit 包装器截断)")
    return "\n".join(lines)


# =====================================================================
# 1. 网络
# =====================================================================
class MLP(nn.Module):
    """两层隐藏层的小型 MLP（tanh 激活），Actor 与 Critic 共用。"""

    def __init__(self, in_dim: int, hidden_size: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_dim)
        # 正交初始化：小增益让初始策略接近确定性，避免一开始就疯狂探索
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.out(x)


class Actor(nn.Module):
    """策略网络：输出高斯分布的均值 mu 与可学习的 log_std。

    标准差 std = exp(log_std) 控制探索强度：
      std 大 -> 采样到的 u 分散 -> 动作多样（探索），但 log_prob 小
      std 小 -> 动作集中在 tanh(mu) 附近（利用）
    log_std 是与状态无关的独立可学习参数（对角高斯）。
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int, init_log_std: float = -0.5):
        super().__init__()
        self.net = MLP(obs_dim, hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.net(obs)                                  # [B, act_dim]
        log_std = self.log_std.expand_as(mu).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std


class Critic(nn.Module):
    """价值网络：输出标量 V(s)。"""

    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.net = MLP(obs_dim, hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # squeeze(-1)：输出 [B] 而不是 [B, 1]。
        # 如果保留 [B, 1]，和 [B] 的 advantage 相减会广播成 [B, B]，这是经典陷阱。
        return self.net(obs).squeeze(-1)


# =====================================================================
# 2. 连续动作策略：tanh 高斯 + 正确的 log_prob
# =====================================================================
class TanhGaussianPolicy(nn.Module):
    """把 Actor 的 (mu, log_std) 变成一个带 tanh 压缩的连续动作分布。

    采样与映射：
        u      ~ Normal(mu, std)          # u 在实轴上，无界
        action = action_bias + action_scale * tanh(u)   # 映射到 env 的动作范围

    其中 action_bias = (high + low) / 2，action_scale = (high - low) / 2，
    完全由 env.action_space 决定，不写死。因为 tanh 的值域是 (-1, 1)，
    所以 action 严格落在 (low, high) 内部，永远合法。

    log_prob 需要 Jacobian 修正（变量替换公式）：
        log pi(action) = log N(u; mu, std) - log |d action / d u|
        log |d action / d u| = sum_d [ log(scale_d) + log(1 - tanh(u_d)^2) ]
    最后一项在动作维度上求和。
    """

    def __init__(
        self,
        obs_dim: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        hidden_size: int,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        low = torch.as_tensor(list(action_low), dtype=torch.float32)
        high = torch.as_tensor(list(action_high), dtype=torch.float32)
        if low.shape != high.shape:
            raise ValueError("action_low / action_high 形状不一致")
        self.actor = Actor(obs_dim, int(low.numel()), hidden_size, init_log_std)
        # 注册为 buffer：随 state_dict 保存，随 .to(device) 迁移
        self.register_buffer("action_bias", (high + low) / 2.0)
        self.register_buffer("action_scale", (high - low) / 2.0)

    # ---------- 分布 ----------
    def distribution(self, obs: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        mu, log_std = self.actor(obs)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std), mu, std

    # ---------- Jacobian ----------
    def log_tanh_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        """返回 log|d action / d u|，形状 [B]。"""
        log_scale = torch.log(self.action_scale).sum()  # 标量：sum_d log(scale_d)
        # 数值稳定写法：log(1 - tanh(u)^2) = 2*log2 - 2u - 2*softplus(-2u)
        # 直接写 torch.log(1 - tanh(u)**2) 在 |u| 稍大时会出现 log(0) = -inf。
        log_sech2 = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        return log_scale + log_sech2.sum(dim=-1)

    def log_prob_from_u(self, dist: torch.distributions.Normal, u: torch.Tensor) -> torch.Tensor:
        """在给定分布下计算 u 对应的**变换后动作**的 log_prob，形状 [B]。

        梯度会经过 dist 的参数 (mu, log_std)，但不会经过 u 本身。
        """
        log_prob_u = dist.log_prob(u).sum(dim=-1)          # 各维求和 -> [B]
        return log_prob_u - self.log_tanh_jacobian(u)      # 减去 Jacobian

    # ---------- 采样 ----------
    def sample(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """采样动作。

        关键：u 先 detach 再重新计算 log_prob。
        这样 Actor 损失里 (log_prob * advantage) 的梯度**只**经过 mu / log_std，
        不会沿着重参数化路径穿回采样噪声。这就是 "把动作样本当作固定样本"。
        （dist.rsample().detach() 与 dist.sample() 在这里等价，
          写 rsample().detach() 是为了把意图显式表达出来。）
        """
        dist, mu, std = self.distribution(obs)
        u = dist.rsample().detach()                        # [B, act_dim]，固定样本
        log_prob = self.log_prob_from_u(dist, u)           # [B]
        action = self.action_bias + self.action_scale * torch.tanh(u)
        return {"action": action, "log_prob": log_prob, "u": u, "mu": mu, "std": std}

    # ---------- 确定性动作（评估用）----------
    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """评估用：取变换后高斯分布的均值，即 bias + scale * tanh(mu)。"""
        mu, _ = self.actor(obs)
        return self.action_bias + self.action_scale * torch.tanh(mu)


# =====================================================================
# 3. Rollout：单环境采样，记录 episode 边界
# =====================================================================
@dataclass
class Rollout:
    """一段 on-policy 数据。所有张量都在 device 上，第一维是时间 T。"""

    obs: torch.Tensor        # [T, obs_dim]
    u: torch.Tensor          # [T, act_dim]  采样得到的 pre-tanh 动作（已 detach）
    log_prob: torch.Tensor   # [T]           采样时的 log_prob（仅用于诊断）
    std: torch.Tensor        # [T, act_dim]  采样时的动作标准差
    reward: torch.Tensor     # [T]           环境**原始**奖励
    terminated: torch.Tensor  # [T] bool      真实终止
    truncated: torch.Tensor   # [T] bool      时间限制截断
    next_obs: torch.Tensor   # [T, obs_dim]  每步 step() 返回的观测 s_{t+1}

    def to(self, device: torch.device) -> "Rollout":
        return Rollout(
            obs=self.obs.to(device),
            u=self.u.to(device),
            log_prob=self.log_prob.to(device),
            std=self.std.to(device),
            reward=self.reward.to(device),
            terminated=self.terminated.to(device),
            truncated=self.truncated.to(device),
            next_obs=self.next_obs.to(device),
        )


class RolloutCollector:
    """持有 "当前观测 / 当前 episode 的累计奖励与长度" 状态。

    因为一个 episode 通常比一段 rollout 长，这些状态必须在多次采样之间保留。
    """

    def __init__(self, env: gym.Env, device: torch.device, seed: Optional[int] = None):
        self.env = env
        self.device = device
        obs, _ = env.reset(seed=seed)
        if seed is not None:
            env.action_space.seed(seed)
        self.obs = np.asarray(obs, dtype=np.float32)
        self.ep_reward = 0.0
        self.ep_length = 0
        # 已完成的 episode：(原始累计奖励, 长度)
        self.finished_episodes: List[Tuple[float, int]] = []

    def collect(self, policy: TanhGaussianPolicy, n_steps: int) -> Rollout:
        obs_l, u_l, logp_l, std_l, rew_l, term_l, trunc_l, nobs_l = ([] for _ in range(8))
        finished_before = len(self.finished_episodes)

        with torch.no_grad():
            for _ in range(n_steps):
                obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                out = policy.sample(obs_t)
                action = out["action"].squeeze(0).cpu().numpy().astype(np.float64)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                obs_l.append(self.obs)
                u_l.append(out["u"].squeeze(0).cpu())
                logp_l.append(out["log_prob"].squeeze(0).cpu())
                std_l.append(out["std"].squeeze(0).cpu())
                rew_l.append(float(reward))
                term_l.append(bool(terminated))
                trunc_l.append(bool(truncated))
                # next_obs 必须在这里存下来：如果这一步是 truncated，
                # 后面我们会 reset，reset 后的观测属于**下一个** episode，
                # 不能拿来给当前 episode bootstrap。
                nobs_l.append(np.asarray(next_obs, dtype=np.float32))

                self.ep_reward += float(reward)
                self.ep_length += 1

                if terminated or truncated:
                    self.finished_episodes.append((self.ep_reward, self.ep_length))
                    next_obs, _ = self.env.reset()
                    self.obs = np.asarray(next_obs, dtype=np.float32)
                    self.ep_reward = 0.0
                    self.ep_length = 0
                else:
                    self.obs = np.asarray(next_obs, dtype=np.float32)

        rollout = Rollout(
            obs=torch.as_tensor(np.stack(obs_l), dtype=torch.float32),
            u=torch.stack(u_l),
            log_prob=torch.stack(logp_l),
            std=torch.stack(std_l),
            reward=torch.as_tensor(rew_l, dtype=torch.float32),
            terminated=torch.as_tensor(term_l, dtype=torch.bool),
            truncated=torch.as_tensor(trunc_l, dtype=torch.bool),
            next_obs=torch.as_tensor(np.stack(nobs_l), dtype=torch.float32),
        )
        self.last_new_episodes = self.finished_episodes[finished_before:]
        return rollout.to(self.device)


# =====================================================================
# 4. n-step 回报目标
# =====================================================================
def compute_nstep_returns(
    rewards: torch.Tensor,      # [T] 已按 reward_scale 缩放
    next_values: torch.Tensor,  # [T] V(s_{t+1})，无梯度
    terminated: torch.Tensor,   # [T] bool
    truncated: torch.Tensor,    # [T] bool
    gamma: float,
) -> torch.Tensor:
    """从后往前递推 n-step 回报 R_t，形状 [T]，结果无梯度（是回归目标）。

        R_t = r_t + gamma * ( 继续递推的 R_{t+1}   若 episode 未结束
                             V(s_{t+1})            若 episode 在此结束 )

    三种边界的处理：
      * terminated : 真实终止 -> 不 bootstrap，链在此截断（V 记 0）
      * truncated  : 时间限制 -> 用截断状态 s_{t+1} 的价值 bootstrap
      * rollout 末尾: 用 s_T 的价值 bootstrap（若末尾是 terminated 则不 bootstrap）

    递推只在 episode 内部进行，绝不跨越 episode 边界。
    """
    T = rewards.shape[0]
    if T == 0:
        return rewards.clone()

    returns = torch.empty_like(rewards)
    zero = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    carry = zero  # 表示 R_{t+1}

    for t in range(T - 1, -1, -1):
        # s_{t+1} 的价值：真实终止时为 0（不 bootstrap），否则用 V(s_{t+1})
        bootstrap = zero if terminated[t] else next_values[t]
        # 只有当 episode 在 t 之后仍在继续、且 t 不是 rollout 末尾时，才沿着 R_{t+1} 递推
        continues = (t < T - 1) and (not bool(terminated[t])) and (not bool(truncated[t]))
        v_next = carry if continues else bootstrap
        returns[t] = rewards[t] + gamma * v_next
        carry = returns[t]

    return returns


# =====================================================================
# 5. 一次 A2C 更新
# =====================================================================
@dataclass
class UpdateStats:
    actor_loss: float
    critic_loss: float
    mean_std: float
    mean_advantage: float
    mean_return_target: float
    grad_norm_actor: float
    grad_norm_critic: float
    mean_log_prob: float


def update_once(
    policy: TanhGaussianPolicy,
    critic: Critic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    rollout: Rollout,
    cfg: "TrainConfig",
) -> UpdateStats:
    """用一段 rollout 做一次（且仅一次）参数更新。"""
    obs, u = rollout.obs, rollout.u
    T = obs.shape[0]

    # --- 价值：需要梯度，供 critic_loss 使用 ---
    values = critic(obs)                                    # [T]

    # --- 回报目标：全程 no_grad，目标是常数 ---
    with torch.no_grad():
        next_values = critic(rollout.next_obs)              # [T]
        scaled_reward = rollout.reward / cfg.reward_scale   # 缩放只影响优化尺度
        returns = compute_nstep_returns(
            scaled_reward, next_values, rollout.terminated, rollout.truncated, cfg.gamma
        )

    # --- 优势：detach，保证 Actor 的梯度不会流进 Critic ---
    advantage = (returns - values.detach())                 # [T]

    # --- Actor 损失：用当前策略重新计算 log_prob，样本 u 是固定的 ---
    dist, mu, std = policy.distribution(obs)
    log_prob = policy.log_prob_from_u(dist, u)              # [T]，梯度只经过 mu / log_std

    adv_actor = advantage
    if cfg.normalize_advantage:
        # 只标准化 Actor 用的优势；Critic 的回归目标 returns 保持原样
        adv_std = adv_actor.std(unbiased=False)
        # adv_std ≈ 0 时（n_steps=1 只有一个样本，或这段 rollout 里优势恰好全相同），
        # 标准化会把优势整体抹成 0，Actor 的梯度随之全为 0 且不报任何错。
        # 这时退回未标准化的优势，保证 Actor 仍能更新。
        if adv_std > 1e-8:
            adv_actor = (adv_actor - adv_actor.mean()) / adv_std

    actor_loss = -(log_prob * adv_actor.detach()).mean()

    if cfg.entropy_coef > 0.0:
        # 注意：这里用的是**基础高斯** Normal(mu, std) 的熵作为近似，
        # 它并不是 tanh 变换后动作分布的精确熵（后者没有闭式解）。
        base_entropy = dist.entropy().sum(dim=-1)           # [T]
        actor_loss = actor_loss - cfg.entropy_coef * base_entropy.mean()

    # --- Critic 损失：均方误差，目标是常数 ---
    critic_loss = F.mse_loss(values, returns.detach())

    # --- 分别反向传播。actor_loss 的计算图里只有 Actor 的参数，critic_loss 只有 Critic 的，
    #     所以两个 loss 可以各自 backward，互不干扰。---
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    grad_norm_actor = float(nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm))
    actor_opt.step()

    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    grad_norm_critic = float(nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm))
    critic_opt.step()

    return UpdateStats(
        actor_loss=float(actor_loss.detach()),
        critic_loss=float(critic_loss.detach()),
        mean_std=float(std.mean().detach()),
        # 乘回 reward_scale：advantage / returns 都在缩放尺度上，但日志（updates.csv 等）
        # 一律报告环境原始奖励单位，好和 episodes.csv / eval.csv 直接对照。
        mean_advantage=float(advantage.mean()) * cfg.reward_scale,
        mean_return_target=float(returns.mean()) * cfg.reward_scale,
        grad_norm_actor=grad_norm_actor,
        grad_norm_critic=grad_norm_critic,
        mean_log_prob=float(log_prob.mean().detach()),
    )


# =====================================================================
# 6. 评估
# =====================================================================
@torch.no_grad()
def evaluate(
    policy: TanhGaussianPolicy,
    env: gym.Env,
    n_episodes: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """用确定性动作（变换后高斯均值）在独立评估环境上跑 n_episodes 个回合。

    不采样、不更新、不影响训练环境。
    """
    rewards, lengths = [], []
    for i in range(n_episodes):
        reset_seed = None if seed is None else seed + i
        obs, _ = env.reset(seed=reset_seed)
        ep_reward, ep_len, done = 0.0, 0, False
        while not done:
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
            action = policy.deterministic_action(obs_t).squeeze(0).cpu().numpy().astype(np.float64)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(ep_len)
    return np.asarray(rewards, dtype=np.float64), np.asarray(lengths, dtype=np.int64)


# =====================================================================
# 7. 模型保存 / 加载
# =====================================================================
def save_checkpoint(
    path: Path,
    policy: TanhGaussianPolicy,
    critic: Critic,
    env_info: Dict[str, object],
    config: Dict[str, object],
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """保存网络权重 + 网络配置 + 动作边界 + 训练配置，保证可独立加载。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "env_info": env_info,      # 含 obs_dim / action_low / action_high，加载时据此重建网络
        "config": config,          # 含 hidden_size 等
        "actor_state_dict": policy.state_dict(),
        "critic_state_dict": critic.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[TanhGaussianPolicy, Critic, Dict, Dict, Dict]:
    """加载模型，返回 (policy, critic, env_info, config, extra)。"""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # 老版本 torch 没有 weights_only 参数
        ckpt = torch.load(path, map_location=device)

    env_info = ckpt["env_info"]
    config = ckpt["config"]
    policy = TanhGaussianPolicy(
        obs_dim=int(env_info["obs_dim"]),
        action_low=env_info["action_low"],
        action_high=env_info["action_high"],
        hidden_size=int(config["hidden_size"]),
    ).to(device)
    policy.load_state_dict(ckpt["actor_state_dict"])
    policy.eval()

    critic = Critic(int(env_info["obs_dim"]), int(config["hidden_size"])).to(device)
    critic.load_state_dict(ckpt["critic_state_dict"])
    critic.eval()

    extra = {k: v for k, v in ckpt.items() if k not in
             ("format_version", "env_info", "config", "actor_state_dict", "critic_state_dict")}
    return policy, critic, env_info, config, extra


# =====================================================================
# 8. 配置
# =====================================================================
@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "auto"
    total_timesteps: int = 1_000_000
    n_steps: int = 32
    gamma: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 3e-3
    hidden_size: int = 64
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.0
    normalize_advantage: bool = True
    reward_scale: float = 100.0
    log_dir: str = "logs/a2c_lunarlander"
    model_path: str = "models/a2c_lunarlander_best.pt"
    eval_interval: int = 20_000
    eval_episodes: int = 10
    log_interval: int = 10
    init_log_std: float = -0.5
    plot: bool = True

    def __post_init__(self) -> None:
        """把非法参数挡在训练开始前，而不是跑到一半崩在取模或空数组上。"""
        if self.n_steps < 1:
            raise ValueError(f"n_steps 必须 >= 1，收到 {self.n_steps}")
        if self.eval_interval < 1:
            raise ValueError(f"eval_interval 必须 >= 1，收到 {self.eval_interval}")
        if self.eval_episodes < 1:
            raise ValueError(f"eval_episodes 必须 >= 1，收到 {self.eval_episodes}")
        if self.log_interval < 1:
            raise ValueError(f"log_interval 必须 >= 1，收到 {self.log_interval}")
        if self.reward_scale <= 0:
            raise ValueError(f"reward_scale 必须 > 0，收到 {self.reward_scale}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda 但当前环境没有可用的 CUDA")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================================
# 9. 日志与绘图
# =====================================================================
class CSVLogger:
    """极简 CSV 记录器，按需写表头。"""

    def __init__(self, path: Path, header: Sequence[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
        self.file.flush()

    def log(self, row: Sequence[object]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def plot_reward_curve(episode_rewards: Sequence[float], out_path: Path, window: int = 100) -> None:
    """训练奖励曲线：浅色原始曲线 + 深色滑动平均。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not episode_rewards:
        return
    rewards = np.asarray(episode_rewards, dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(np.arange(1, len(rewards) + 1), rewards, color="#b9c6d6", linewidth=0.9, label="episode reward (raw)")
    if len(rewards) >= 2:
        w = min(window, len(rewards))
        kernel = np.ones(w) / w
        smooth = np.convolve(rewards, kernel, mode="valid")
        ax.plot(np.arange(w, len(rewards) + 1), smooth, color="#1f4e79", linewidth=2.0,
                label=f"moving average ({w})")
    ax.axhline(200, color="#c0504d", linestyle="--", linewidth=1.2, label="solved threshold (200)")
    ax.set_xlabel("episode")
    ax.set_ylabel("raw episode reward")
    ax.set_title("A2C (n-step) on LunarLander-v3 continuous")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =====================================================================
# 10. 训练
# =====================================================================
def train(cfg: TrainConfig) -> Dict[str, object]:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg.model_path)
    final_model_path = model_path.with_name(model_path.stem.replace("_best", "") + "_final" + model_path.suffix)

    env = make_env()
    env_info = env_info_from(env)
    print("=" * 78)
    print(describe_env(env_info))
    print("=" * 78)
    print(f"device = {device} | seed = {cfg.seed} | torch = {torch.__version__}")

    policy = TanhGaussianPolicy(
        obs_dim=int(env_info["obs_dim"]),
        action_low=env_info["action_low"],
        action_high=env_info["action_high"],
        hidden_size=cfg.hidden_size,
        init_log_std=cfg.init_log_std,
    ).to(device)
    critic = Critic(int(env_info["obs_dim"]), cfg.hidden_size).to(device)

    actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    collector = RolloutCollector(env, device, seed=cfg.seed)
    eval_env = make_env()

    ep_logger = CSVLogger(
        log_dir / "episodes.csv",
        ["episode", "env_steps", "raw_episode_reward", "episode_length", "moving_avg_100"],
    )
    upd_logger = CSVLogger(
        log_dir / "updates.csv",
        ["update", "env_steps", "actor_loss", "critic_loss", "action_std",
         "mean_advantage", "mean_return_target", "grad_norm_actor", "grad_norm_critic", "mean_log_prob"],
    )
    eval_logger = CSVLogger(
        log_dir / "eval.csv",
        ["env_steps", "mean_raw_reward", "std_raw_reward", "min_raw_reward", "max_raw_reward", "mean_length"],
    )

    with open(log_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump({"train_config": asdict(cfg), "env_info": env_info}, fh, indent=2, ensure_ascii=False)

    all_episode_rewards: List[float] = []
    best_eval_reward = -float("inf")
    env_steps = 0
    update_idx = 0
    next_eval_step = cfg.eval_interval
    start_time = time.time()
    last_print = start_time

    print(f"开始训练: total_timesteps={cfg.total_timesteps}, n_steps={cfg.n_steps}, gamma={cfg.gamma}")
    print("-" * 78)

    try:
        while env_steps < cfg.total_timesteps:
            rollout = collector.collect(policy, cfg.n_steps)
            env_steps += cfg.n_steps
            update_idx += 1

            stats = update_once(policy, critic, actor_opt, critic_opt, rollout, cfg)

            # 记录本段 rollout 内结束的 episode（原始奖励，不做任何缩放）
            for ep_reward, ep_len in collector.last_new_episodes:
                all_episode_rewards.append(ep_reward)
                recent = all_episode_rewards[-100:]
                ep_logger.log([
                    len(all_episode_rewards), env_steps, round(ep_reward, 3), ep_len,
                    round(float(np.mean(recent)), 3),
                ])

            upd_logger.log([
                update_idx, env_steps,
                round(stats.actor_loss, 6), round(stats.critic_loss, 6), round(stats.mean_std, 6),
                round(stats.mean_advantage, 6), round(stats.mean_return_target, 6),
                round(stats.grad_norm_actor, 6), round(stats.grad_norm_critic, 6),
                round(stats.mean_log_prob, 6),
            ])

            # 定期评估
            if env_steps >= next_eval_step:
                eval_rewards, eval_lengths = evaluate(
                    policy, eval_env, cfg.eval_episodes, device, seed=cfg.seed + 10_000
                )
                mean_r, std_r = float(eval_rewards.mean()), float(eval_rewards.std())
                eval_logger.log([
                    env_steps, round(mean_r, 3), round(std_r, 3),
                    round(float(eval_rewards.min()), 3), round(float(eval_rewards.max()), 3),
                    round(float(eval_lengths.mean()), 2),
                ])
                if mean_r > best_eval_reward:
                    best_eval_reward = mean_r
                    save_checkpoint(
                        model_path, policy, critic, env_info, asdict(cfg),
                        extra={"env_steps": env_steps, "eval_mean_reward": mean_r,
                               "eval_std_reward": std_r, "kind": "best"},
                    )
                print(f"  [eval @ {env_steps:>9,d} steps] mean={mean_r:8.2f} std={std_r:7.2f} "
                      f"best={best_eval_reward:8.2f}")
                next_eval_step += cfg.eval_interval

            # 定期打印进度
            now = time.time()
            if update_idx % cfg.log_interval == 0 or now - last_print > 30.0:
                last_print = now
                avg100 = float(np.mean(all_episode_rewards[-100:])) if all_episode_rewards else float("nan")
                sps = env_steps / max(now - start_time, 1e-6)
                print(f"step {env_steps:>9,d}/{cfg.total_timesteps:,d} | ep {len(all_episode_rewards):>5d} | "
                      f"avg100 {avg100:8.2f} | actor_loss {stats.actor_loss:9.4f} | "
                      f"critic_loss {stats.critic_loss:10.4f} | std {stats.mean_std:.3f} | {sps:6.1f} steps/s")

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，提前结束训练，保存当前模型。")

    # 保存最终模型
    save_checkpoint(
        final_model_path, policy, critic, env_info, asdict(cfg),
        extra={"env_steps": env_steps, "kind": "final"},
    )
    ep_logger.close()
    upd_logger.close()
    eval_logger.close()

    curve_path = log_dir / "reward_curve.png"
    if cfg.plot:
        try:
            plot_reward_curve(all_episode_rewards, curve_path)
            print(f"奖励曲线已保存: {curve_path}")
        except Exception as exc:  # 绘图失败不应影响训练结果
            print(f"[警告] 绘图失败: {exc}")
            curve_path = None
    else:
        curve_path = None

    env.close()
    eval_env.close()

    total_time = time.time() - start_time
    final_avg = float(np.mean(all_episode_rewards[-100:])) if all_episode_rewards else float("nan")
    summary = {
        "total_timesteps": env_steps,
        "episodes": len(all_episode_rewards),
        "final_moving_avg_100": final_avg,
        "best_eval_mean_reward": best_eval_reward if best_eval_reward > -float("inf") else None,
        "wall_clock_seconds": round(total_time, 1),
        "best_model_path": str(model_path) if model_path.exists() else None,
        "final_model_path": str(final_model_path),
        "reward_curve_path": str(curve_path) if curve_path else None,
        "log_dir": str(log_dir),
    }
    print("-" * 78)
    print("训练结束")
    for key, value in summary.items():
        print(f"  {key:26s}: {value}")
    if env_steps < 200_000:
        print("\n[注意] 本次训练步数远低于完整训练预算，只能验证流程是否跑通，")
        print("       不足以证明策略已经学会着陆。请用默认参数做完整训练。")
    return summary


# =====================================================================
# 11. 评估 / 渲染
# =====================================================================
def run_evaluate(model_path: Path, episodes: int, seed: int, device_name: str) -> Dict[str, object]:
    device = resolve_device(device_name)
    policy, _critic, env_info, _config, extra = load_checkpoint(model_path, device)
    print(describe_env(env_info))
    env = make_env()
    rewards, lengths = evaluate(policy, env, episodes, device, seed=seed)
    env.close()
    result = {
        "model_path": str(model_path),
        "trained_env_steps": extra.get("env_steps"),
        "episodes": episodes,
        "mean_raw_reward": float(rewards.mean()),
        "std_raw_reward": float(rewards.std()),
        "min_raw_reward": float(rewards.min()),
        "max_raw_reward": float(rewards.max()),
        "mean_episode_length": float(lengths.mean()),
        "per_episode_rewards": [round(float(r), 3) for r in rewards],
    }
    print(f"评估 {episodes} 个回合 (确定性动作, 原始环境奖励):")
    print(f"  平均奖励 = {result['mean_raw_reward']:.2f} +/- {result['std_raw_reward']:.2f}")
    print(f"  最小/最大 = {result['min_raw_reward']:.2f} / {result['max_raw_reward']:.2f}")
    print(f"  平均回合长度 = {result['mean_episode_length']:.1f}")
    return result


def run_render(
    model_path: Path,
    episodes: int,
    seed: int,
    device_name: str,
    video_path: Optional[Path],
    fps: int,
) -> None:
    device = resolve_device(device_name)
    policy, _critic, _env_info, _config, _extra = load_checkpoint(model_path, device)

    # 有 --video-path 时用 rgb_array 离屏渲染（无图形界面也能用）；
    # 否则用 human 打开窗口。
    render_mode = "rgb_array" if video_path else "human"
    env = make_env(render_mode=render_mode)
    frames: List[np.ndarray] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_reward, ep_len = False, 0.0, 0
        while not done:
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
            action = policy.deterministic_action(obs_t).squeeze(0).cpu().numpy().astype(np.float64)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
            if video_path:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        print(f"  episode {ep + 1}: reward = {ep_reward:.2f}, length = {ep_len}")
    env.close()

    if video_path and frames:
        import imageio
        video_path.parent.mkdir(parents=True, exist_ok=True)
        # macro_block_size=None: 保持环境原始分辨率 (600x400)，否则 ffmpeg 会自动填充到 16 的倍数
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)
        print(f"视频已保存: {video_path} ({len(frames)} 帧, {fps} fps)")


# =====================================================================
# 12. verify：环境 / 形状 / 数值 自检
# =====================================================================
def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def run_verify(seed: int, device_name: str) -> bool:
    device = resolve_device(device_name)
    print("=" * 78)
    print(f"自检开始 (device={device}, seed={seed})")
    print("=" * 78)
    all_ok = True
    set_seed(seed)

    # ---------- 1) 环境确实是连续动作空间 ----------
    print("\n[1] 环境与动作空间")
    env = make_env()
    info = env_info_from(env)
    print(describe_env(info))
    all_ok &= _check("动作空间是 Box", isinstance(env.action_space, gym.spaces.Box))
    all_ok &= _check("env.unwrapped.continuous is True", bool(getattr(env.unwrapped, "continuous", False)))
    all_ok &= _check("观测维度 > 0", info["obs_dim"] > 0, f"obs_dim={info['obs_dim']}")
    all_ok &= _check("动作维度 > 0", info["action_dim"] > 0, f"action_dim={info['action_dim']}")

    # ---------- 2) 动作始终在合法范围内 ----------
    print("\n[2] 采样动作的合法性 (500 步随机策略)")
    policy = TanhGaussianPolicy(
        info["obs_dim"], info["action_low"], info["action_high"], hidden_size=64
    ).to(device)
    obs, _ = env.reset(seed=seed)
    lows = np.asarray(info["action_low"], dtype=np.float64)
    highs = np.asarray(info["action_high"], dtype=np.float64)
    in_range, env_accepted = True, True
    with torch.no_grad():
        for _ in range(500):
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
            action = policy.sample(obs_t)["action"].squeeze(0).cpu().numpy().astype(np.float64)
            if not np.all(action >= lows) or not np.all(action <= highs):
                in_range = False
            try:
                obs, _r, term, trunc, _ = env.step(action)
            except Exception as exc:
                env_accepted = False
                print(f"       env.step 抛错: {exc}")
                break
            if term or trunc:
                obs, _ = env.reset()
    all_ok &= _check("动作始终落在 action_space 上下界内", in_range)
    all_ok &= _check("环境接受所有采样动作", env_accepted)
    env.close()

    # ---------- 3) 形状 ----------
    print("\n[3] 张量形状")
    T, obs_dim, act_dim = 8, info["obs_dim"], info["action_dim"]
    obs_b = torch.randn(T, obs_dim, device=device)
    out = policy.sample(obs_b)
    dist, mu, std = policy.distribution(obs_b)
    logp = policy.log_prob_from_u(dist, out["u"])
    all_ok &= _check("mu/std 形状 [T, act_dim]", tuple(mu.shape) == (T, act_dim), str(tuple(mu.shape)))
    all_ok &= _check("action 形状 [T, act_dim]", tuple(out["action"].shape) == (T, act_dim))
    all_ok &= _check("u 形状 [T, act_dim]", tuple(out["u"].shape) == (T, act_dim))
    all_ok &= _check("log_prob 形状 [T]", tuple(logp.shape) == (T,), str(tuple(logp.shape)))
    all_ok &= _check("log_prob 在动作维度上求和(非 [T, act_dim])", logp.dim() == 1)
    all_ok &= _check("u 已 detach (固定样本)", not out["u"].requires_grad)
    all_ok &= _check("log_prob 保留梯度", logp.requires_grad)

    critic = Critic(obs_dim, 64).to(device)
    values = critic(obs_b)
    all_ok &= _check("Critic 输出形状 [T] (不是 [T,1])", tuple(values.shape) == (T,), str(tuple(values.shape)))

    # 广播陷阱检查：adv 与 logp 相乘不能变成 [T, T]
    fake_adv = torch.randn(T, device=device)
    prod = (logp * fake_adv).shape
    all_ok &= _check("log_prob * advantage 形状仍为 [T]", tuple(prod) == (T,), str(tuple(prod)))

    # ---------- 4) log_prob 的数值正确性（与数值微分/朴素公式对比）----------
    print("\n[4] log_prob 的 Jacobian 修正")
    u_test = torch.tensor([[-0.3, 0.7], [8.0, -12.0], [0.0, 0.0]], device=device)  # 含极端值测数值稳定
    jac = policy.log_tanh_jacobian(u_test)
    naive = torch.log(policy.action_scale).sum() + torch.log(
        torch.clamp(1.0 - torch.tanh(u_test) ** 2, min=1e-30)
    ).sum(dim=-1)
    all_ok &= _check("稳定公式与朴素公式在温和区域一致",
                     torch.allclose(jac[:1], naive[:1], atol=1e-5), f"{jac[:1].tolist()} vs {naive[:1].tolist()}")
    all_ok &= _check("极端 u 下无 NaN/Inf", bool(torch.isfinite(jac).all()), str(jac.tolist()))
    # 单调性检查：tanh 饱和时 Jacobian 项应该很大（密度被压缩），而不是 -inf 或 nan
    all_ok &= _check("饱和区 Jacobian 项显著小于温和区", float(jac[1].mean()) < float(jac[0].mean()))
    # 与采样 log_prob 一致性：同一个 u 用同一分布算两次应完全相同
    lp_a = policy.log_prob_from_u(dist, out["u"])
    lp_b = policy.log_prob_from_u(dist, out["u"])
    all_ok &= _check("log_prob 可重复", torch.allclose(lp_a, lp_b))

    # ---------- 5) n-step 回报的边界语义 ----------
    print("\n[5] n-step 回报的 terminated / truncated 处理")
    gamma = 0.5
    rew = torch.tensor([1.0, 1.0, 1.0], device=device)
    nv = torch.full((3,), 9.0, device=device)   # V(s_1)=V(s_2)=V(s_3)=9

    r_term = compute_nstep_returns(rew, nv, torch.tensor([False, False, True], device=device),
                                   torch.tensor([False, False, False], device=device), gamma)
    exp_term = torch.tensor([1.75, 1.5, 1.0], device=device)
    all_ok &= _check("terminated 不 bootstrap (期望 [1.75, 1.5, 1.0])",
                     torch.allclose(r_term, exp_term, atol=1e-6), str(r_term.tolist()))

    r_trunc_end = compute_nstep_returns(rew, nv, torch.tensor([False, False, False], device=device),
                                        torch.tensor([False, False, True], device=device), gamma)
    exp_trunc_end = torch.tensor([2.875, 3.75, 5.5], device=device)
    all_ok &= _check("rollout 末尾 truncated 用 V(s_T) bootstrap (期望 [2.875, 3.75, 5.5])",
                     torch.allclose(r_trunc_end, exp_trunc_end, atol=1e-6), str(r_trunc_end.tolist()))

    r_trunc_mid = compute_nstep_returns(rew, nv, torch.tensor([False, False, False], device=device),
                                        torch.tensor([False, True, False], device=device), gamma)
    exp_trunc_mid = torch.tensor([3.75, 5.5, 5.5], device=device)
    all_ok &= _check("中途 truncated 不跨越 episode (期望 [3.75, 5.5, 5.5])",
                     torch.allclose(r_trunc_mid, exp_trunc_mid, atol=1e-6), str(r_trunc_mid.tolist()))

    # n_steps=1 应退化为一步 TD
    r_td = compute_nstep_returns(rew[:1], nv[:1], torch.tensor([False], device=device),
                                 torch.tensor([False], device=device), gamma)
    all_ok &= _check("n_steps=1 退化为一步 TD: R = r + gamma*V(s')",
                     abs(float(r_td[0]) - (1.0 + gamma * 9.0)) < 1e-6, str(r_td.tolist()))

    # ---------- 6) 真实 rollout + 一次更新：无 NaN/Inf、参数会变、Critic 不被 Actor 影响 ----------
    print("\n[6] 真实 rollout + 一次 A2C 更新")
    env2 = make_env()
    collector = RolloutCollector(env2, device, seed=seed)
    cfg = TrainConfig(seed=seed, n_steps=8, gamma=0.99, hidden_size=64)
    actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    rollout = collector.collect(policy, cfg.n_steps)
    all_ok &= _check("rollout.obs 形状 [T, obs_dim]", tuple(rollout.obs.shape) == (cfg.n_steps, obs_dim))
    all_ok &= _check("rollout.reward 形状 [T]", tuple(rollout.reward.shape) == (cfg.n_steps,))
    all_ok &= _check("rollout.next_obs 形状 [T, obs_dim]", tuple(rollout.next_obs.shape) == (cfg.n_steps, obs_dim))

    actor_before = [p.detach().clone() for p in policy.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]
    stats = update_once(policy, critic, actor_opt, critic_opt, rollout, cfg)
    print(f"       actor_loss={stats.actor_loss:.6f} critic_loss={stats.critic_loss:.6f} "
          f"std={stats.mean_std:.4f} adv={stats.mean_advantage:.4f} target={stats.mean_return_target:.4f}")

    finite = all(math.isfinite(v) for v in
                 (stats.actor_loss, stats.critic_loss, stats.mean_std, stats.mean_advantage,
                  stats.grad_norm_actor, stats.grad_norm_critic))
    all_ok &= _check("损失 / 统计量无 NaN 或 Inf", finite)
    grads_finite = all(torch.isfinite(p.grad).all() for p in list(policy.parameters()) + list(critic.parameters())
                       if p.grad is not None)
    all_ok &= _check("梯度无 NaN 或 Inf", bool(grads_finite))
    actor_changed = any(not torch.equal(a, b) for a, b in zip(actor_before, policy.parameters()))
    critic_changed = any(not torch.equal(a, b) for a, b in zip(critic_before, critic.parameters()))
    all_ok &= _check("Actor 参数已更新", actor_changed)
    all_ok &= _check("Critic 参数已更新", critic_changed)

    # Actor 的梯度不应依赖 Critic 的图：单独 backward actor_loss 时 critic 的 grad 应为空
    critic_opt.zero_grad(set_to_none=True)
    actor_opt.zero_grad(set_to_none=True)
    dist2, _mu2, _std2 = policy.distribution(rollout.obs)
    lp2 = policy.log_prob_from_u(dist2, rollout.u)
    with torch.no_grad():
        v2 = critic(rollout.obs)
        ret2 = compute_nstep_returns(rollout.reward, critic(rollout.next_obs),
                                     rollout.terminated, rollout.truncated, cfg.gamma)
    adv2 = (ret2 - v2).detach()      # 完全 detach：Actor 的图里没有 Critic
    (-(lp2 * adv2).mean()).backward()
    critic_has_grad = any(p.grad is not None for p in critic.parameters())
    actor_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    all_ok &= _check("Actor loss 反传不会给 Critic 参数留下梯度", not critic_has_grad)
    all_ok &= _check("Actor 参数确实收到梯度", actor_has_grad)
    actor_opt.zero_grad(set_to_none=True)
    env2.close()

    # ---------- 7) 保存 / 加载 / 评估 ----------
    print("\n[7] 模型保存、加载与评估")
    tmp_model = Path(tempfile.gettempdir()) / f"_verify_a2c_{seed}.pt"
    save_checkpoint(tmp_model, policy, critic, info, asdict(cfg), extra={"env_steps": 1234})
    all_ok &= _check("模型文件已写出", tmp_model.exists(), str(tmp_model))

    policy2, critic2, info2, cfg2, extra2 = load_checkpoint(tmp_model, device)
    all_ok &= _check("加载后 obs_dim / action 边界一致",
                     info2["obs_dim"] == info["obs_dim"] and info2["action_low"] == info["action_low"]
                     and info2["action_high"] == info["action_high"])
    all_ok &= _check("加载后 hidden_size 一致", int(cfg2["hidden_size"]) == cfg.hidden_size)
    all_ok &= _check("加载后 extra 元信息保留", extra2.get("env_steps") == 1234)
    with torch.no_grad():
        same = torch.allclose(policy.deterministic_action(obs_b), policy2.deterministic_action(obs_b))
    all_ok &= _check("加载后 Actor 前向输出与保存前一致", bool(same))

    eval_env = make_env()
    ev_rewards, ev_lengths = evaluate(policy2, eval_env, 2, device, seed=seed)
    all_ok &= _check("评估可执行且返回合理形状", ev_rewards.shape == (2,) and ev_lengths.shape == (2,),
                     f"rewards={np.round(ev_rewards, 2).tolist()}")
    eval_env.close()
    tmp_model.unlink(missing_ok=True)

    # ---------- 8) 小型数值例子 ----------
    print("\n[8] 小型数值例子 (gamma=0.9, reward_scale=1)")
    r_demo = torch.tensor([1.0, 0.0], device=device)
    v_demo = torch.tensor([1.0, 2.0], device=device)          # V(s_0), V(s_1)
    nv_demo = torch.tensor([2.0, 0.5], device=device)         # V(s_1)=2.0, V(s_2)=0.5 (末尾 truncated)
    ret_demo = compute_nstep_returns(r_demo, nv_demo,
                                     torch.tensor([False, False], device=device),
                                     torch.tensor([False, True], device=device), 0.9)
    adv_demo = ret_demo - v_demo
    norm_demo = (adv_demo - adv_demo.mean()) / adv_demo.std(unbiased=False)
    lp_demo = torch.tensor([-0.5, -2.0], device=device)
    actor_loss_demo = -(lp_demo * norm_demo.detach()).mean()
    critic_loss_demo = F.mse_loss(v_demo, ret_demo.detach())
    print(f"       rewards        = {r_demo.tolist()}")
    print(f"       V(s_t)         = {v_demo.tolist()}")
    print(f"       V(s_t+1)       = {nv_demo.tolist()}   (t=1 处 truncated, 用 V(s_2) bootstrap)")
    print(f"       R_t            = {[round(x, 6) for x in ret_demo.tolist()]}"
          f"   (R_1 = 0 + 0.9*0.5 = 0.45; R_0 = 1 + 0.9*0.45 = 1.405)")
    print(f"       A_t = R_t - V  = {[round(x, 6) for x in adv_demo.tolist()]}")
    print(f"       A_t 标准化     = {[round(x, 6) for x in norm_demo.tolist()]}")
    print(f"       log_prob(a_t)  = {lp_demo.tolist()}")
    print(f"       actor_loss     = -mean(logp * A_norm) = {float(actor_loss_demo):.6f}")
    print(f"       critic_loss    = mean((V - R)^2)      = {float(critic_loss_demo):.6f}")
    all_ok &= _check("数值例子结果与手算一致",
                     abs(float(ret_demo[1]) - 0.45) < 1e-6 and abs(float(ret_demo[0]) - 1.405) < 1e-6
                     and abs(float(actor_loss_demo) - (-0.75)) < 1e-6
                     and abs(float(critic_loss_demo) - 1.2832625) < 1e-3)

    print("\n" + "=" * 78)
    print(f"自检结果: {'全部通过' if all_ok else '存在失败项，请查看上面的 [FAIL]'}")
    print("=" * 78)
    return bool(all_ok)


# =====================================================================
# 13. 命令行
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="n-step Advantage Actor-Critic (A2C) on LunarLander-v3 (continuous)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=0, help="Python / NumPy / PyTorch / 环境 随机种子")
        p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="自动检测 CUDA；cpu 始终可用")

    # ---- train ----
    p_train = sub.add_parser("train", help="训练并保存日志、曲线与模型",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p_train)
    p_train.add_argument("--total-timesteps", type=int, default=1_000_000,
                         help="训练预算，按环境交互步数计。默认 100 万步 (单环境 CPU 约 1-2 小时)")
    p_train.add_argument("--n-steps", type=int, default=32,
                         help="每段 rollout 的长度，也决定 n-step 回报的 n。设为 1 即一步 TD Actor-Critic")
    p_train.add_argument("--gamma", type=float, default=0.99,
                         help="折扣因子。LunarLander 回合最长 1000 步，0.99~0.999 都常用")
    p_train.add_argument("--actor-lr", type=float, default=1e-4, help="Actor 学习率")
    p_train.add_argument("--critic-lr", type=float, default=3e-3,
                         help="Critic 学习率。通常比 Actor 大，让价值估计更快跟上")
    p_train.add_argument("--hidden-size", type=int, default=64, help="两个网络隐藏层宽度")
    p_train.add_argument("--max-grad-norm", type=float, default=0.5, help="梯度裁剪阈值")
    p_train.add_argument("--entropy-coef", type=float, default=0.0,
                         help="熵正则系数。默认 0（基础版本不启用）。启用时使用基础高斯熵的近似")
    p_train.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True,
                         help="是否标准化 Actor 用的优势（Critic 的回报目标不标准化）")
    p_train.add_argument("--reward-scale", type=float, default=100.0,
                         help="正数。奖励除以它后再进入回报目标与优势，用于改善优化尺度；"
                              "日志与评估始终报告环境原始奖励")
    p_train.add_argument("--init-log-std", type=float, default=-0.5, help="log_std 初始值 (std≈0.61)")
    p_train.add_argument("--log-dir", type=str, default="logs/a2c_lunarlander", help="日志目录")
    p_train.add_argument("--model-path", type=str, default="models/a2c_lunarlander_best.pt",
                         help="最佳模型保存路径（最终模型自动保存为 *_final.pt）")
    p_train.add_argument("--eval-interval", type=int, default=20_000, help="每多少环境步评估一次")
    p_train.add_argument("--eval-episodes", type=int, default=10, help="每次评估的回合数")
    p_train.add_argument("--log-interval", type=int, default=10, help="每多少次更新打印一次进度")
    p_train.add_argument("--no-plot", dest="plot", action="store_false", default=True,
                         help="不生成奖励曲线图")

    # ---- evaluate ----
    p_eval = sub.add_parser("evaluate", help="加载模型并评估",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p_eval)
    p_eval.add_argument("--model-path", type=str, required=True, help="模型文件路径")
    p_eval.add_argument("--episodes", type=int, default=20, help="评估回合数")
    p_eval.add_argument("--json-out", type=str, default=None, help="可选：把评估结果写成 JSON")

    # ---- render ----
    p_render = sub.add_parser("render", help="可视化着陆过程 / 导出视频",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p_render)
    p_render.add_argument("--model-path", type=str, required=True, help="模型文件路径")
    p_render.add_argument("--episodes", type=int, default=5, help="演示回合数")
    p_render.add_argument("--video-path", type=str, default=None,
                          help="保存 mp4 的路径。给出时用 rgb_array 离屏渲染，无图形界面也能用")
    p_render.add_argument("--fps", type=int, default=50, help="视频帧率")

    # ---- verify ----
    p_verify = sub.add_parser("verify", help="环境 / 形状 / 边界 / 数值自检",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p_verify)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        try:
            cfg = TrainConfig(
                seed=args.seed, device=args.device, total_timesteps=args.total_timesteps,
                n_steps=args.n_steps, gamma=args.gamma, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                hidden_size=args.hidden_size, max_grad_norm=args.max_grad_norm,
                entropy_coef=args.entropy_coef, normalize_advantage=args.normalize_advantage,
                reward_scale=args.reward_scale, log_dir=args.log_dir, model_path=args.model_path,
                eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
                log_interval=args.log_interval, init_log_std=args.init_log_std, plot=args.plot,
            )
        except ValueError as exc:
            raise SystemExit(f"参数错误：{exc}") from None
        train(cfg)
        return 0

    if args.command == "evaluate":
        result = run_evaluate(Path(args.model_path), args.episodes, args.seed, args.device)
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"结果已写入: {args.json_out}")
        return 0

    if args.command == "render":
        run_render(Path(args.model_path), args.episodes, args.seed, args.device,
                   Path(args.video_path) if args.video_path else None, args.fps)
        return 0

    if args.command == "verify":
        return 0 if run_verify(args.seed, args.device) else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
