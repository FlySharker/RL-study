"""Q-learning solution for the FrozenLake environment.

Install dependency:
    pip install gymnasium[toy-text]

Run:
    python examples/from_scratch/q_learning_frozenlake.py
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # Fallback for older installations.
    try:
        import gym  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with:\n"
            "    pip install -r requirements.txt\n"
            "or:\n"
            "    pip install gymnasium[toy-text]"
        ) from exc


ACTION_SYMBOLS = {
    0: "<",  # left
    1: "v",  # down
    2: ">",  # right
    3: "^",  # up
}


def reset_env(env: Any, seed: int | None = None) -> int:
    result = env.reset(seed=seed)
    if isinstance(result, tuple):
        observation, _info = result
        return int(observation)
    return int(result)


def step_env(env: Any, action: int) -> tuple[int, float, bool]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, _info = result
        done = terminated or truncated
    else:
        observation, reward, done, _info = result
    return int(observation), float(reward), bool(done)


def choose_action(
    q_table: np.ndarray,
    state: int,
    epsilon: float,
    action_count: int,
) -> int:
    if random.random() < epsilon:
        return random.randrange(action_count)
    return int(np.argmax(q_table[state]))


def train_q_learning(
    env: Any,
    episodes: int,
    max_steps: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
    min_epsilon: float,
    epsilon_decay: float,
    seed: int,
) -> np.ndarray:
    state_count = env.observation_space.n
    action_count = env.action_space.n
    q_table = np.zeros((state_count, action_count), dtype=np.float64)

    random.seed(seed)
    np.random.seed(seed)

    for episode in range(episodes):
        state = reset_env(env, seed=seed + episode)

        for _step in range(max_steps):
            action = choose_action(q_table, state, epsilon, action_count)
            next_state, reward, done = step_env(env, action)

            best_next_action_value = np.max(q_table[next_state])
            td_target = reward + discount_factor * best_next_action_value
            td_error = td_target - q_table[state, action]
            q_table[state, action] += learning_rate * td_error

            state = next_state
            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table


def evaluate_policy(env: Any, q_table: np.ndarray, episodes: int, max_steps: int) -> float:
    wins = 0

    for episode in range(episodes):
        state = reset_env(env, seed=10_000 + episode)

        for _step in range(max_steps):
            action = int(np.argmax(q_table[state]))
            state, reward, done = step_env(env, action)

            if done:
                wins += int(reward > 0)
                break

    return wins / episodes


def print_policy(env: Any, q_table: np.ndarray) -> None:
    desc = env.unwrapped.desc.astype(str)
    best_actions = np.argmax(q_table, axis=1)

    print("\nLearned policy:")
    state = 0
    for row in desc:
        symbols = []
        for cell in row:
            if cell in {"H", "G"}:
                symbols.append(cell)
            else:
                symbols.append(ACTION_SYMBOLS[int(best_actions[state])])
            state += 1
        print(" ".join(symbols))


def watch_policy(
    q_table: np.ndarray,
    slippery: bool,
    max_steps: int,
    delay: float,
    episodes: int,
) -> None:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=slippery,
        render_mode="human",
    )

    for episode in range(episodes):
        state = reset_env(env, seed=20_000 + episode)

        for _step in range(max_steps):
            action = int(np.argmax(q_table[state]))
            state, _reward, done = step_env(env, action)
            env.render()
            time.sleep(delay)

            if done:
                break

        time.sleep(1.0)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Q-learning agent on FrozenLake."
    )
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=1_000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--slippery",
        action="store_true",
        help="Use the stochastic FrozenLake environment.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Show a render window after training.",
    )
    parser.add_argument(
        "--watch-episodes",
        type=int,
        default=1,
        help="Number of rendered evaluation episodes.",
    )
    parser.add_argument(
        "--watch-delay",
        type=float,
        default=0.5,
        help="Seconds to wait between rendered steps.",
    )
    args = parser.parse_args()

    env = gym.make("FrozenLake-v1", is_slippery=args.slippery)

    q_table = train_q_learning(
        env=env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    success_rate = evaluate_policy(
        env=env,
        q_table=q_table,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )

    np.set_printoptions(precision=3, suppress=True)
    print("Q-table:")
    print(q_table)
    print_policy(env, q_table)
    print(f"\nSuccess rate: {success_rate:.1%}")

    env.close()

    if args.watch:
        watch_policy(
            q_table=q_table,
            slippery=args.slippery,
            max_steps=args.max_steps,
            delay=args.watch_delay,
            episodes=args.watch_episodes,
        )


if __name__ == "__main__":
    main()
