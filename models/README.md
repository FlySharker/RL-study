# 本地模型

这里保存手写算法的 `.pt` 权重、SB3 的 `.zip` 模型以及配套的 `*_vecnormalize.pkl` 统计量。产物默认被 Git 忽略；本文件用于说明目录用途。

REINFORCE 训练时将权重、曲线和 CSV 一起保存到 `logs/reinforce_pendulum/`，可通过 `--output-dir` 修改。

MuJoCo PPO 评估应同时使用对应训练运行产生的模型与归一化统计。比如：

```bash
python examples/sb3/sb3_ppo_halfcheetah.py --no-train --load-path models/sb3_ppo_halfcheetah.zip --vecnormalize-path models/sb3_ppo_halfcheetah_vecnormalize.pkl
```

相对路径以仓库根目录为工作目录。不要把不同训练运行的统计文件混用。
