"""Shared Stable-Baselines3 PPO runner for Gymnasium MuJoCo tasks."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

if os.environ.get("MUJOCO_GL") in {"egl", "osmesa"} and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]

try:
    import gymnasium as gym
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with:\n"
        "    pip install -r requirements.txt"
    ) from exc


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_normalize_env = self.model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            vec_normalize_env.save(self.save_path)
            if self.verbose:
                print(f"Saved VecNormalize stats to: {self.save_path}")
        return True


def activation_from_name(name: str) -> type[nn_module]:
    if name.lower() == "relu":
        return th.nn.ReLU
    if name.lower() == "tanh":
        return th.nn.Tanh
    raise SystemExit("Unsupported activation. Use 'tanh' or 'relu'.")


nn_module = th.nn.Module


def make_single_env(env_id: str, seed: int, render_mode: str | None = None) -> Any:
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except Exception as exc:
        message = str(exc).lower()
        if "mujoco" in message:
            raise SystemExit(
                f"{env_id} needs MuJoCo dependencies. Install them with:\n"
                "    pip install -r requirements.txt\n"
                "or:\n"
                "    pip install gymnasium[mujoco]"
            ) from exc
        raise

    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_train_env(args: argparse.Namespace) -> VecNormalize:
    try:
        env = make_vec_env(
            args.env_id,
            n_envs=args.n_envs,
            seed=args.seed,
            monitor_dir=args.log_dir,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "mujoco" in message:
            raise SystemExit(
                f"{args.env_id} needs MuJoCo dependencies. Install them with:\n"
                "    pip install -r requirements.txt\n"
                "or:\n"
                "    pip install gymnasium[mujoco]"
            ) from exc
        raise

    return VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=args.clip_obs,
        gamma=args.gamma,
    )


def make_eval_env(
    args: argparse.Namespace,
    seed: int,
    render_mode: str | None = None,
    stats_path: str | None = None,
) -> VecNormalize:
    env = DummyVecEnv([lambda: make_single_env(args.env_id, seed, render_mode=render_mode)])
    vec_stats_path = Path(stats_path or args.vecnormalize_path)

    if vec_stats_path.exists():
        env = VecNormalize.load(vec_stats_path, env)
    else:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=args.clip_obs,
            gamma=args.gamma,
        )

    env.training = False
    env.norm_reward = False
    return env


def pick_stats_path_for_model(args: argparse.Namespace, model_path: Path) -> str:
    sibling_stats = model_path.parent / "vecnormalize.pkl"
    if sibling_stats.exists():
        return str(sibling_stats)
    return args.vecnormalize_path


def train_model(args: argparse.Namespace) -> PPO:
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vecnormalize_path).parent.mkdir(parents=True, exist_ok=True)
    train_env = make_train_env(args)
    eval_env = make_eval_env(args, args.seed + 10_000)

    save_vecnormalize_callback = SaveVecNormalizeCallback(args.best_vecnormalize_path)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.best_model_dir,
        log_path=args.log_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=save_vecnormalize_callback,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        target_kl=args.target_kl,
        policy_kwargs={
            "activation_fn": activation_from_name(args.activation_fn),
            "net_arch": dict(
                pi=[args.hidden_size, args.hidden_size],
                vf=[args.hidden_size, args.hidden_size],
            ),
        },
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
        )
    except KeyboardInterrupt:
        interrupted_path = args.save_path.replace(".zip", "_interrupted.zip")
        interrupted_stats_path = args.vecnormalize_path.replace(".pkl", "_interrupted.pkl")
        model.save(interrupted_path)
        train_env.save(interrupted_stats_path)
        print(f"\nInterrupted. Saved current model to: {interrupted_path}")
        print(f"Saved VecNormalize stats to: {interrupted_stats_path}")
        raise

    model.save(args.save_path)
    train_env.save(args.vecnormalize_path)
    print(f"Saved final model to: {args.save_path}")
    print(f"Saved final VecNormalize stats to: {args.vecnormalize_path}")

    train_env.close()
    eval_env.close()

    best_model_path = Path(args.best_model_dir) / "best_model.zip"
    if best_model_path.exists():
        print(f"Loading best evaluated model from: {best_model_path}")
        best_eval_env = make_eval_env(args, args.seed + 20_000, stats_path=args.best_vecnormalize_path)
        return PPO.load(best_model_path, env=best_eval_env, device=args.device)

    return model


def load_model(args: argparse.Namespace) -> PPO:
    model_path = Path(args.load_path or args.save_path)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    stats_path = pick_stats_path_for_model(args, model_path)
    env = make_eval_env(args, args.seed + 30_000, stats_path=stats_path)
    return PPO.load(model_path, env=env, device=args.device)


def evaluate_model(model: PPO, args: argparse.Namespace) -> None:
    model_path = Path(args.load_path or args.save_path)
    stats_path = pick_stats_path_for_model(args, model_path)
    env = make_eval_env(args, args.seed + 40_000, stats_path=stats_path)
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


def watch_model(model: PPO, args: argparse.Namespace) -> None:
    model_path = Path(args.load_path or args.save_path)
    stats_path = pick_stats_path_for_model(args, model_path)
    env = make_eval_env(args, args.seed + 50_000, render_mode="human", stats_path=stats_path)
    obs = env.reset()

    for episode in range(args.watch_episodes):
        episode_reward = 0.0

        for _step in range(args.max_steps):
            action, _state = model.predict(obs, deterministic=not args.stochastic_watch)
            obs, reward, done, _info = env.step(action)
            episode_reward += float(reward[0])
            env.render()
            time.sleep(args.watch_delay)

            if bool(done[0]):
                obs = env.reset()
                break

        print(f"Watch episode {episode + 1} reward: {episode_reward:.2f}")
        time.sleep(1.0)

    env.close()


def record_video(model: PPO, args: argparse.Namespace) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit(
            "Missing video dependency. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from exc

    model_path = Path(args.load_path or args.save_path)
    stats_path = pick_stats_path_for_model(args, model_path)
    output_dir = Path(args.video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_eval_env(args, args.seed + 80_000, render_mode="rgb_array", stats_path=stats_path)
    obs = env.reset()

    for episode in range(args.video_episodes):
        episode_reward = 0.0
        video_path = output_dir / f"{args.env_id.lower()}_episode_{episode + 1}.mp4"

        with imageio.get_writer(video_path, fps=args.video_fps) as writer:
            for _step in range(args.max_steps):
                frame = env.render()
                if isinstance(frame, list):
                    frame = frame[0]
                if frame is not None:
                    writer.append_data(frame)

                action, _state = model.predict(obs, deterministic=not args.stochastic_watch)
                obs, reward, done, _info = env.step(action)
                episode_reward += float(reward[0])

                if bool(done[0]):
                    obs = env.reset()
                    break

        print(f"Recorded episode {episode + 1} reward: {episode_reward:.2f}")
        print(f"Saved video to: {video_path}")

    env.close()


def infer_single_action(model: PPO, args: argparse.Namespace, state_text: str, deterministic: bool) -> None:
    values = [float(item.strip()) for item in state_text.split(",")]
    raw_obs = np.asarray(values, dtype=np.float32)

    probe_env = make_single_env(args.env_id, args.seed + 60_000)
    expected_dim = int(probe_env.observation_space.shape[0])
    probe_env.close()

    if len(values) != expected_dim:
        raise SystemExit(f"{args.env_id} state must contain exactly {expected_dim} numbers.")

    model_path = Path(args.load_path or args.save_path)
    stats_path = pick_stats_path_for_model(args, model_path)
    if Path(stats_path).exists():
        env = make_eval_env(args, args.seed + 70_000, stats_path=stats_path)
        obs = env.normalize_obs(raw_obs.reshape(1, -1))
        env.close()
    else:
        obs = raw_obs

    action, _state = model.predict(obs, deterministic=deterministic)
    print(f"State length: {len(values)}")
    print(f"Action: {np.asarray(action).squeeze().tolist()}")


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    env_id: str,
    total_timesteps: int,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    use_sde: bool,
    sde_sample_freq: int,
    hidden_size: int,
    activation_fn: str,
    eval_freq: int,
    eval_episodes: int,
    max_steps: int,
    save_prefix: str,
) -> argparse.ArgumentParser:
    parser.add_argument("--env-id", type=str, default=env_id)
    parser.add_argument("--total-timesteps", type=int, default=total_timesteps)
    parser.add_argument("--n-envs", type=int, default=n_envs)
    parser.add_argument("--n-steps", type=int, default=n_steps)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--n-epochs", type=int, default=n_epochs)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--gamma", type=float, default=gamma)
    parser.add_argument("--gae-lambda", type=float, default=gae_lambda)
    parser.add_argument("--clip-range", type=float, default=clip_range)
    parser.add_argument("--ent-coef", type=float, default=ent_coef)
    parser.add_argument("--vf-coef", type=float, default=vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=max_grad_norm)
    parser.add_argument("--target-kl", type=float, default=target_kl)
    parser.add_argument("--use-sde", action="store_true", default=use_sde)
    parser.add_argument("--no-use-sde", action="store_false", dest="use_sde")
    parser.add_argument("--sde-sample-freq", type=int, default=sde_sample_freq)
    parser.add_argument("--hidden-size", type=int, default=hidden_size)
    parser.add_argument("--activation-fn", type=str, default=activation_fn)
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--eval-freq", type=int, default=eval_freq)
    parser.add_argument("--eval-episodes", type=int, default=eval_episodes)
    parser.add_argument("--max-steps", type=int, default=max_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default=f"models/{save_prefix}.zip")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--vecnormalize-path", type=str, default=f"models/{save_prefix}_vecnormalize.pkl")
    parser.add_argument("--best-model-dir", type=str, default=f"logs/{save_prefix}/best")
    parser.add_argument("--best-vecnormalize-path", type=str, default=f"logs/{save_prefix}/best/vecnormalize.pkl")
    parser.add_argument("--log-dir", type=str, default=f"logs/{save_prefix}")
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-episodes", type=int, default=1)
    parser.add_argument("--watch-delay", type=float, default=0.01)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-dir", type=str, default=f"logs/{save_prefix}/videos")
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--stochastic-watch", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--state", type=str, default=None)
    return parser


def run_ppo_task(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()

    if args.load_path or args.no_train:
        model = load_model(args)
        print(f"Loaded model from: {args.load_path or args.save_path}")
    else:
        model = train_model(args)

    if args.state is not None:
        infer_single_action(model, args, args.state, deterministic=not args.stochastic_eval)
        return

    evaluate_model(model, args)

    if args.record_video:
        record_video(model, args)

    if args.watch:
        watch_model(model, args)
