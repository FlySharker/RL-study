# RL Study

强化学习学习与实验代码集合，包含 NumPy / PyTorch 手写实现，以及 Stable-Baselines3（SB3）训练示例。环境使用 Gymnasium，覆盖离散控制、连续控制和 MuJoCo 任务。

## 目录

```text
rl_study/
├── examples/
│   ├── from_scratch/                 # 手写算法与推理入口，全部代码平铺
│   │   ├── actor_critic_lunarlander.py
│   │   ├── dqn_cartpole.py
│   │   ├── infer_cartpole.py
│   │   ├── ppo_lunarlander.py
│   │   ├── q_learning_frozenlake.py
│   │   ├── reinforce_pendulum.py
│   │   └── sac_bipedalwalker.py
│   └── sb3/                          # SB3 示例与 MuJoCo PPO 公共模块
│       ├── sb3_ppo_ant.py
│       ├── sb3_ppo_halfcheetah.py
│       ├── sb3_ppo_hopper.py
│       ├── sb3_ppo_humanoid.py
│       ├── sb3_ppo_humanoidstandup.py
│       ├── sb3_ppo_mujoco_common.py
│       ├── sb3_ppo_pendulum.py
│       ├── sb3_ppo_pusher.py
│       ├── sb3_ppo_walker2d.py
│       ├── sb3_sac_ant.py
│       └── sb3_sac_bipedalwalker.py
├── scripts/check_project.py          # 语法检查及可选命令入口检查
├── models/                           # 本地模型与归一化统计，Git 忽略产物
├── logs/                             # 训练日志，运行时生成，Git 忽略
└── requirements.txt                  # 全部示例依赖的最低版本要求
```

## 创建环境与安装依赖（Anaconda）

先安装 Anaconda。Windows 用户可打开 Anaconda Prompt，进入仓库根目录，再创建并激活 Python 3.11 环境：

```bash
conda create -n rl_study python=3.11 -y
conda activate rl_study
```

在已激活的 conda 环境内安装依赖，先安装 SWIG，再安装其余依赖：

```bash
python -m pip install --upgrade pip
python -m pip install swig
python -m pip install -r requirements.txt
```

安装后可检查核心依赖是否可以导入：

```bash
python -c "import torch, gymnasium, stable_baselines3; print(torch.__version__, gymnasium.__version__, stable_baselines3.__version__)"
```

以后打开新的终端时，先运行 `conda activate rl_study` 再执行项目命令。退出环境可运行 `conda deactivate`。

SWIG 先于 Box2D 依赖安装。PyTorch 的 CPU / CUDA 安装方式可按运行机器调整；代码无需强制使用 GPU。依赖文件声明最低版本，并非锁定环境；不同版本、硬件和随机种子下的训练结果可能不同。

全部示例统一使用根目录的 `requirements.txt`。SB3 示例开启 `--progress-bar` 时还需要安装 `tqdm` 和 `rich`。

## 快速开始

**下列命令统一从仓库根目录运行。** 相对模型路径和日志路径均相对于当前工作目录；脚本迁移后，参数名称和算法实现保持原有方式。

```bash
# 查看参数，不启动训练
python examples/from_scratch/dqn_cartpole.py --help

# Q-learning：FrozenLake，无需预训练模型
python examples/from_scratch/q_learning_frozenlake.py

# DQN：训练 CartPole 并保存到 models/dqn_cartpole.pt
python examples/from_scratch/dqn_cartpole.py

# 加载 DQN 模型评估；添加 --watch 可打开窗口
python examples/from_scratch/infer_cartpole.py --model-path models/dqn_cartpole.pt

# A2C：环境、张量形状、回报边界、梯度与模型读写自检
python examples/from_scratch/actor_critic_lunarlander.py verify

# SB3 PPO：训练 Pendulum，默认模型保存到 models/
python examples/sb3/sb3_ppo_pendulum.py
```

训练命令可能运行较长时间。完整参数通过各脚本的 `--help` 查看；A2C 也支持 `train --help`、`evaluate --help` 和 `render --help`。A2C 的算法讲解与自检逻辑保留在代码中。

## 示例索引

| 实现 | 算法 / 任务 | 入口（相对于仓库根目录） |
|---|---|---|
| 手写 | Q-learning / FrozenLake | `examples/from_scratch/q_learning_frozenlake.py` |
| 手写 | DQN / CartPole | `examples/from_scratch/dqn_cartpole.py` |
| 手写 | DQN 模型推理 | `examples/from_scratch/infer_cartpole.py` |
| 手写 | REINFORCE / Pendulum | `examples/from_scratch/reinforce_pendulum.py` |
| 手写 | PPO / LunarLander 离散动作 | `examples/from_scratch/ppo_lunarlander.py` |
| 手写 | A2C / LunarLander 连续动作 | `examples/from_scratch/actor_critic_lunarlander.py` |
| 手写 | SAC / BipedalWalker | `examples/from_scratch/sac_bipedalwalker.py` |
| SB3 | PPO / Ant-v5 | `examples/sb3/sb3_ppo_ant.py` |
| SB3 | PPO / HalfCheetah-v5 | `examples/sb3/sb3_ppo_halfcheetah.py` |
| SB3 | PPO / Hopper-v5 | `examples/sb3/sb3_ppo_hopper.py` |
| SB3 | PPO / Humanoid-v5 | `examples/sb3/sb3_ppo_humanoid.py` |
| SB3 | PPO / HumanoidStandup-v5 | `examples/sb3/sb3_ppo_humanoidstandup.py` |
| SB3 | PPO / Pendulum-v1 | `examples/sb3/sb3_ppo_pendulum.py` |
| SB3 | PPO / Pusher-v5 | `examples/sb3/sb3_ppo_pusher.py` |
| SB3 | PPO / Walker2d-v5 | `examples/sb3/sb3_ppo_walker2d.py` |
| SB3 | SAC / Ant-v5 | `examples/sb3/sb3_sac_ant.py` |
| SB3 | SAC / BipedalWalker-v3 | `examples/sb3/sb3_sac_bipedalwalker.py` |
| SB3 | MuJoCo PPO 公共模块 | `examples/sb3/sb3_ppo_mujoco_common.py` |

`examples/sb3/sb3_ppo_mujoco_common.py` 是共享模块，不是独立训练入口。脚本保留相邻模块导入，请按示例直接执行文件。

## 训练与评估

评估需要先训练或准备好对应模型。以下命令从仓库根目录执行：

```bash
# REINFORCE：模型、CSV 和曲线默认进入 logs/reinforce_pendulum/
python examples/from_scratch/reinforce_pendulum.py
python examples/from_scratch/reinforce_pendulum.py --mode eval --model-path logs/reinforce_pendulum/reinforce_pendulum.pth

# 手写 PPO：LunarLander 离散动作
python examples/from_scratch/ppo_lunarlander.py
python examples/from_scratch/ppo_lunarlander.py --no-train --load-path models/ppo_lunarlander.pt

# 手写 SAC：BipedalWalker
python examples/from_scratch/sac_bipedalwalker.py --device cpu
python examples/from_scratch/sac_bipedalwalker.py --no-train --load-path models/sac_bipedalwalker.pt

# A2C：LunarLander 连续动作
python examples/from_scratch/actor_critic_lunarlander.py train
python examples/from_scratch/actor_critic_lunarlander.py evaluate --model-path models/a2c_lunarlander_best.pt --episodes 20
python examples/from_scratch/actor_critic_lunarlander.py render --model-path models/a2c_lunarlander_best.pt --episodes 5

# SB3 PPO：MuJoCo，模型与归一化统计应来自同一次训练
python examples/sb3/sb3_ppo_halfcheetah.py --device cpu
python examples/sb3/sb3_ppo_halfcheetah.py --no-train --load-path models/sb3_ppo_halfcheetah.zip --vecnormalize-path models/sb3_ppo_halfcheetah_vecnormalize.pkl

# SB3 SAC
python examples/sb3/sb3_sac_ant.py --device cpu
python examples/sb3/sb3_sac_bipedalwalker.py --device cpu
```

DQN 推理、手写 PPO / SAC 和 SB3 示例可追加 `--watch` 打开可视化窗口。REINFORCE 评估使用 `--render`；A2C 使用 `render` 子命令。

## 模型与日志

| 产物 | 默认位置 |
|---|---|
| DQN、手写 PPO / SAC、A2C 模型 | `models/` |
| SB3 最终模型、MuJoCo PPO 归一化统计 | `models/` |
| SB3 评估最优模型与评估日志 | `logs/<脚本名>/` |
| A2C 的 CSV、配置和奖励曲线 | `logs/a2c_lunarlander/` |
| REINFORCE 模型、CSV 和奖励曲线 | `logs/reinforce_pendulum/` |

不同实验应指定独立保存路径和日志目录，避免覆盖已有结果。例如：

```bash
python examples/from_scratch/actor_critic_lunarlander.py train --total-timesteps 64 --n-steps 32 --eval-interval 64 --eval-episodes 1 --log-dir logs/a2c_smoke --model-path models/a2c_smoke_best.pt --no-plot
```

该短命令仅用于验证流程，不代表训练收敛。更多模型目录说明见 [models/README.md](models/README.md)。

支持 TensorBoard 的示例可通过 `--tensorboard-log logs/tensorboard/<实验名>` 开启记录，再运行 `tensorboard --logdir logs/tensorboard` 查看。

## 本地验证

```bash
# 仅使用 Python 标准库，检查源码语法
python scripts/check_project.py

# 安装依赖后，额外逐个检查所有示例的 --help 入口
python scripts/check_project.py --cli
```

这些检查不表示算法已经收敛。训练效果需要结合独立评估、多随机种子和实际曲线判断。

## AI 使用说明

本项目使用 AI 工具辅助代码整理、路径调整、README 编写和运行检查。文中的“手写实现”指直接使用 NumPy / PyTorch 编写算法的实现方式。

AI 辅助生成或修改的代码、注释和算法解释可能存在错误。学习和修改时，应结合源码、算法资料与实际运行结果核对；涉及训练效果的结论，应以真实日志、独立评估和多随机种子实验为依据。语法检查、命令入口检查和短流程验证只说明对应流程可运行，不能证明算法已经收敛。
