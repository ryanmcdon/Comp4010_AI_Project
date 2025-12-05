import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import numpy as np
from mlagents_envs.base_env import ActionTuple

from featurizer import featurize_tabular_state

# Minimal discrete action space expected by the Unity behavior
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE_CW = 3
ACTION_DROP = 4
ACTIONS = [ACTION_LEFT, ACTION_RIGHT, ACTION_ROTATE_CW, ACTION_DROP]


@dataclass
class QLearningConfig:
    episodes: int = 500
    max_steps: int = 500
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    seed: Optional[int] = None
    save_path: Optional[str] = None
    load_path: Optional[str] = None


def _heights_and_holes(board: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute column heights and total holes for a 20x10 board.
    """
    board = np.asarray(board).reshape((20, 10))
    rows, cols = board.shape
    heights = np.zeros(cols, dtype=int)
    holes = 0
    for c in range(cols):
        col = board[:, c]
        filled = np.where(col != 0)[0]
        if filled.size == 0:
            heights[c] = 0
            continue
        top = filled[0]
        heights[c] = rows - top
        holes += int(np.sum(col[top:] == 0))
    return heights, holes


def compute_shaped_reward(board: np.ndarray, locked_flag: int, extra_value: int) -> float:
    """
    Reward shaped by board quality:
      + line/extra bonus (assumed non-negative extra_value)
      - penalty for max height and holes
      + small bonus when a piece locks to encourage progress
    """
    heights, holes = _heights_and_holes(board)
    max_height = float(np.max(heights))
    lines_bonus = max(float(extra_value), 0.0)
    hole_penalty = 0.1 * holes
    height_penalty = 0.05 * max_height
    lock_bonus = 0.05 if locked_flag else 0.0
    step_penalty = 0.01  # small living cost
    return lines_bonus + lock_bonus - hole_penalty - height_penalty - step_penalty


def epsilon_greedy_action(q_table: Dict[Tuple[int, ...], np.ndarray], state: Tuple[int, ...], epsilon: float) -> int:
    """
    Choose an action using epsilon-greedy over the tabular Q-values.
    """
    if np.random.random() < epsilon or state not in q_table:
        return np.random.choice(len(ACTIONS))
    return int(np.argmax(q_table[state]))


def ensure_state(q_table: Dict[Tuple[int, ...], np.ndarray], state: Tuple[int, ...]) -> np.ndarray:
    """
    Ensure a state exists in the Q-table; initialize to zeros if new.
    """
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS), dtype=np.float32)
    return q_table[state]


def parse_env_observation(obs: Any) -> Tuple[np.ndarray, int, int, int]:
    """
    Unpack the observation tensor coming from Unity ML-Agents.
    The environment returns the first 200 values as a 20x10 grid, followed by:
        - piece id
        - locked flag (piece locked on last step)
        - extra integer (e.g., lines cleared)
    """
    flat = np.asarray(obs).flatten()
    if flat.shape[0] < 203:
        raise ValueError("Observation must contain at least 203 values (200 grid + 3 extras).")
    board = flat[:200].reshape((20, 10))
    piece_id = int(flat[200])
    locked_flag = int(flat[201])
    extra_val = int(flat[202])
    return board, piece_id, locked_flag, extra_val


def q_learning_train(env, behavior_name: str, config: QLearningConfig) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Tabular Q-learning trainer for the Tetris environment with 20x10 grid observations.

    The environment follows the ML-Agents API:
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        env.set_actions(behavior_name, action_tuple)
        env.step()

    Returns:
        q_table: dict mapping featurized state tuples to action-value arrays.
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    q_table: Dict[Tuple[int, ...], np.ndarray] = {}
    if config.load_path:
        try:
            with open(config.load_path, "rb") as f:
                q_table = pickle.load(f)
            print(f"Loaded Q-table from {config.load_path} with {len(q_table)} states")
        except FileNotFoundError:
            print(f"Load path {config.load_path} not found. Starting fresh.")

    epsilon = config.epsilon_start

    for episode in range(config.episodes):
        env.reset()
        # Get the initial decision steps
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(decision_steps) == 0:
            raise RuntimeError("No decision steps available after reset.")
        obs = decision_steps.obs[0]
        board, piece_id, locked_flag, extra_val = parse_env_observation(obs)
        state = featurize_tabular_state(board, piece_id, locked_flag, extra_val)
        ensure_state(q_table, state)
        total_reward = 0.0

        for step in range(config.max_steps):
            action_idx = epsilon_greedy_action(q_table, state, epsilon)
            action_val = ACTIONS[action_idx]

            action_tuple = ActionTuple(discrete=np.array([[action_val]], dtype=np.int32))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                # Terminal: use last known observation from terminal_steps.obs if available
                obs_next = terminal_steps.obs[0] if len(terminal_steps.obs) > 0 else obs
                env_reward = terminal_steps.reward[0] if hasattr(terminal_steps.reward, "__getitem__") else terminal_steps.reward
                done = True
            elif len(decision_steps) > 0:
                obs_next = decision_steps.obs[0]
                env_reward = decision_steps.reward[0] if hasattr(decision_steps.reward, "__getitem__") else decision_steps.reward
                done = False
            else:
                # No observations returned; treat as zero-reward no-op
                obs_next = obs
                env_reward = 0.0
                done = False

            next_board, next_piece, next_locked, next_extra = parse_env_observation(obs_next)
            shaped_reward = compute_shaped_reward(next_board, next_locked, next_extra)
            reward = float(env_reward) + shaped_reward
            total_reward += reward

            next_state = featurize_tabular_state(next_board, next_piece, next_locked, next_extra)
            next_q = ensure_state(q_table, next_state)
            current_q = ensure_state(q_table, state)

            td_target = reward + (0 if done else config.gamma * float(np.max(next_q)))
            td_error = td_target - current_q[action_idx]
            current_q[action_idx] += config.alpha * td_error

            state = next_state
            if done:
                break

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        print(f"Episode {episode+1}/{config.episodes} | steps={step+1} | total_reward={total_reward:.2f} | epsilon={epsilon:.3f}")

    if config.save_path:
        with open(config.save_path, "wb") as f:
            pickle.dump(q_table, f)
        print(f"Saved Q-table with {len(q_table)} states to {config.save_path}")

    return q_table


def run_greedy_policy(env, behavior_name: str, q_table: Dict[Tuple[int, ...], np.ndarray], episodes: int = 10, max_steps: int = 500):
    """
    Evaluate a trained Q-table greedily using the ML-Agents API.
    """
    for ep in range(episodes):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(decision_steps) == 0:
            raise RuntimeError("No decision steps available after reset.")
        obs = decision_steps.obs[0]
        board, piece_id, locked_flag, extra_val = parse_env_observation(obs)
        state = featurize_tabular_state(board, piece_id, locked_flag, extra_val)
        total_reward = 0.0

        for step in range(max_steps):
            q_values = ensure_state(q_table, state)
            action_idx = int(np.argmax(q_values))
            action_val = ACTIONS[action_idx]

            action_tuple = ActionTuple(discrete=np.array([[action_val]], dtype=np.int32))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                obs_next = terminal_steps.obs[0] if len(terminal_steps.obs) > 0 else obs
                env_reward = terminal_steps.reward[0] if hasattr(terminal_steps.reward, "__getitem__") else terminal_steps.reward
                done = True
            elif len(decision_steps) > 0:
                obs_next = decision_steps.obs[0]
                env_reward = decision_steps.reward[0] if hasattr(decision_steps.reward, "__getitem__") else decision_steps.reward
                done = False
            else:
                obs_next = obs
                env_reward = 0.0
                done = False

            board, piece_id, locked_flag, extra_val = parse_env_observation(obs_next)
            shaped_reward = compute_shaped_reward(board, locked_flag, extra_val)
            total_reward += float(env_reward) + shaped_reward
            state = featurize_tabular_state(board, piece_id, locked_flag, extra_val)

            if done:
                break
        print(f"[EVAL] Episode {ep+1}/{episodes} total_reward={total_reward:.2f}")


def save_q_table(q_table: Dict[Tuple[int, ...], np.ndarray], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Saved Q-table with {len(q_table)} states to {path}")


def load_q_table(path: str) -> Dict[Tuple[int, ...], np.ndarray]:
    with open(path, "rb") as f:
        table = pickle.load(f)
    print(f"Loaded Q-table with {len(table)} states from {path}")
    return table


def train_and_optionally_eval(env, behavior_name: str, config: Optional[QLearningConfig] = None, eval_episodes: int = 0):
    """
    Convenience entrypoint:
      - trains with the provided config (or defaults)
      - optionally runs a greedy evaluation
    """
    config = config or QLearningConfig()
    q_table = q_learning_train(env, behavior_name, config)
    if eval_episodes > 0:
        run_greedy_policy(env, behavior_name, q_table, episodes=eval_episodes)
    return q_table

