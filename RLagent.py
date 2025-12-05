from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import time
import os
from featurizer import featurize_board, featurize_board_5x5, featurize_board_4x4

#TODO create a policy matrix randomly that is updated with the reward from randomAgent

# Define possible actions and corresponding integer mappings
ACTION_MOVE_LEFT = 1
ACTION_MOVE_RIGHT = 2
ACTION_ROTATE = 3
ACTION_DROP = 4
ACTION_FAST_DROP = 5


# List of all possible actions as integers
possible_actions = [
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_ROTATE,
    ACTION_DROP,
]


#A random agent
def randomAgent(env, possible_actions, behavior_name, gama = 0.99, epsilon = 0.1 , max_episodes = 400, n_bins=1000):
    # Policy matrix initialization: shape (n_bins, actionspace=len(possible_actions))
    # Using featurize_board to map board states to n_bins discrete states
    # Using float32 to reduce memory usage (half the size of float64)
    policy_matrix = np.random.rand(n_bins, len(possible_actions)).astype(np.float32)
    print("Policy matrix shape:", policy_matrix.shape)
    # ------------------------------------------------- 
    # MAIN LOOP
    # -------------------------------------------------
    for episode in range(max_episodes):
        print(f"\n========= EPISODE {episode} START =========")
        print("Policy matrix:", policy_matrix)
        env.reset()
        terminated = False

        while not terminated:
            
            # Get decision + terminal steps from Unity
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # -------------------------------------------------
            # If agent needs an action
            # -------------------------------------------------
            if len(decision_steps) > 0:
                obs = decision_steps.obs[0]  # Get observation tensor
                agent_ids = decision_steps.agent_id  # usually 1 agent

                board, current_piece_id, reward1, reward2 = parse_observation(obs)
                # print("Board:", board)
                print("Current Piece ID:", current_piece_id)
                # Use new featurizer instead of old contouring functions
                state_idx, has_piece = featurize_board(board, piece_id=current_piece_id, n_bins=n_bins)
                print("State index:", state_idx, "Has piece:", has_piece)
                print("-" * 40)

                

                # Select a random action from possible_actions
                action_idx = np.random.randint(0, len(possible_actions))
                action_val = possible_actions[action_idx]
                    
                # Update the policy matrix based on the reward received for the action
                
                # This is similar to a value update but not a full RL algorithm
                if len(terminal_steps) > 0:
                    # Use the true reward at the end of the episode
                    reward = terminal_steps.reward[0] if hasattr(terminal_steps.reward, "__getitem__") else terminal_steps.reward
                else:
                    # Use intermediate reward if available
                    reward = reward1

                # Use action_idx directly (no mirroring needed with new featurizer)
                canonical_action = action_idx

                # Basic update rule: nudges the value estimate for this state-action pair toward the received reward
                # Learning rate alpha is set arbitrarily small
                alpha = 0.1
                policy_matrix[state_idx, canonical_action] = policy_matrix[state_idx, canonical_action] + alpha * (reward - policy_matrix[state_idx, canonical_action])

                action_tuple = ActionTuple(
                    discrete=np.array([[action_val]])
                )
                env.set_actions(behavior_name, action_tuple)

            # -------------------------------------------------
            # Step environment
            # -------------------------------------------------
            env.step()

            # -------------------------------------------------
            # Check if episode ended
            # -------------------------------------------------
            if len(terminal_steps) > 0:
                reward = terminal_steps.reward
                print("[DONE] Episode ended with reward:", reward)
                terminated = True
    return reward


def save_policy_matrix(policy_matrix, filepath):
    """
    Save the policy matrix to a file using numpy's save function.
    
    Args:
        policy_matrix: The policy matrix to save, shape (n_bins, len(possible_actions))
        filepath: Path to the file where the matrix will be saved (e.g., 'policy_matrix.npy')
    """
    np.save(filepath, policy_matrix)
    print(f"Policy matrix saved to {filepath}")


def load_policy_matrix(filepath, n_bins=None, n_actions=None):
    """
    Load the policy matrix from a file.
    
    Args:
        filepath: Path to the file containing the saved matrix
        n_bins: Expected number of bins (optional, for validation)
        n_actions: Expected number of actions (optional, for validation)
    
    Returns:
        policy_matrix: The loaded policy matrix, or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist. Returning None.")
        return None
    
    policy_matrix = np.load(filepath)
    print(f"Policy matrix loaded from {filepath}")
    print(f"Loaded matrix shape: {policy_matrix.shape}")
    
    # Validate dimensions if provided
    if n_bins is not None and policy_matrix.shape[0] != n_bins:
        print(f"Warning: Loaded matrix has {policy_matrix.shape[0]} bins, expected {n_bins}")
    if n_actions is not None and policy_matrix.shape[1] != n_actions:
        print(f"Warning: Loaded matrix has {policy_matrix.shape[1]} actions, expected {n_actions}")
    
    return policy_matrix


# An epsilon-greedy agent that uses the policy matrix to select actions
def epsilonGreedyAgent(env, possible_actions, behavior_name, gamma=0.99, epsilon=0.01, move_before_drop=40, max_episodes=400, n_bins=1000, load_from_file=None, save_to_file=None, featurizer=featurize_board):
    """
    Epsilon-greedy agent that selects actions based on the policy matrix.
    With probability epsilon, selects a random action (exploration).
    With probability (1-epsilon), selects the best action according to the policy matrix (exploitation).
    
    Args:
        env: Unity environment instance
        possible_actions: List of possible action values
        behavior_name: Name of the behavior in the environment
        gamma: Discount factor (default: 0.99)
        epsilon: Exploration rate (default: 0.1)
        move_before_drop: Maximum number of moves before forcing a drop action (default: 8)
        max_episodes: Maximum number of episodes to train (default: 400)
        n_bins: Number of bins for state space discretization (default: 1000)
        load_from_file: Optional path to load an existing policy matrix from file
        save_to_file: Optional path to save the policy matrix after training
        featurizer: Function to featurize the board state (default: featurize_board)
    
    Returns:
        policy_matrix: The optimized policy matrix after training, shape (n_bins, len(possible_actions))
    """
    # Policy matrix initialization: try to load from file, otherwise create new
    if load_from_file is not None:
        policy_matrix = load_policy_matrix(load_from_file, n_bins=n_bins, n_actions=len(possible_actions))
        if policy_matrix is None:
            # If loading failed, initialize randomly
            print("Initializing new policy matrix (load failed)")
            policy_matrix = np.random.rand(n_bins, len(possible_actions)).astype(np.float32)
        elif policy_matrix.shape != (n_bins, len(possible_actions)):
            # If dimensions don't match, initialize new matrix
            print(f"Loaded matrix dimensions don't match. Expected ({n_bins}, {len(possible_actions)}), got {policy_matrix.shape}")
            print("Initializing new policy matrix")
            policy_matrix = np.random.rand(n_bins, len(possible_actions)).astype(np.float32)
    else:
        # Policy matrix initialization: shape (n_bins, actionspace=len(possible_actions))
        # Using featurize_board to map board states to n_bins discrete states
        # Using float32 to reduce memory usage (half the size of float64)
        policy_matrix = np.random.rand(n_bins, len(possible_actions)).astype(np.float32)
    
    # print("Policy matrix shape:", policy_matrix.shape)
    print(f"Using epsilon-greedy policy with epsilon={epsilon}")
    
    # ------------------------------------------------- 
    # MAIN LOOP
    # -------------------------------------------------
    for episode in range(max_episodes):
        print(f"\n========= EPISODE {episode} START =========")
        #print("Policy matrix:", policy_matrix)
        env.reset()
        terminated = False
        # List to store state-action pairs until a reward is returned
        state_action_history = []
        currentpiece = -1
        move_number = 0
        moves_in_episode = 0  # Track number of moves in current episode
        
        while not terminated:
            # -------------------------------------------------
            # Step environment
            # -------------------------------------------------
            env.step()
            # Get decision + terminal steps from Unity
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            obs_after = decision_steps.obs[0]
            board, current_piece_id, reward1, reward2 = parse_observation(obs_after)
            # -------------------------------------------------
            # Check reward and update previous actions if reward is returned
            # -------------------------------------------------
            
            # Get the reward after the step
            reward = 0
            if len(terminal_steps) > 0:
                # Use the true reward at the end of the episode
                reward = terminal_steps.reward[0] if hasattr(terminal_steps.reward, "__getitem__") else terminal_steps.reward
                reward = reward1
                # Mark episode as terminated
                terminated = True
                print(f"[DONE] Episode ended with reward: {reward}")
            elif len(decision_steps) > 0:
                # Get reward from observation after step
                
                reward = reward1
            else:
                reward = 0
            
            # If we got a non-zero reward, update all previous state-action pairs
            if reward != 0 and len(state_action_history) > 0 and len(decision_steps) > 0:
                #print("Placement reward:", reward)
                #print(f"Reward returned: {reward}. Updating {len(state_action_history)} previous state-action pairs.")
                alpha = 0.01
                # Update all previous state-action pairs with this reward
                # Optimized: avoid creating temporary array in subtraction
                indices = np.array(state_action_history, dtype=np.int32)
                states = indices[:, 0]
                actions = indices[:, 1]
                current_values = policy_matrix[states, actions]
                policy_matrix[states, actions] = current_values + alpha * (reward - current_values)
                # Clear the history after updating
                state_action_history = []
                
            
            # -------------------------------------------------
            # If take an action
            # -------------------------------------------------
            if len(decision_steps) > 0:
                # agent_ids = decision_steps.agent_id  # usually 1 agent
 
                # print("reward:", reward1)
                if currentpiece == -1:
                    currentpiece = current_piece_id
                else:
                    if currentpiece != current_piece_id:
                        currentpiece = current_piece_id
                        move_number = 0
                        moves_in_episode = 0  # Reset move counter when new piece appears
                    elif move_number <= move_before_drop:
                        move_number += 1
                    else:
                        move_number = -1    #Error
                
                #print("Board:", board)
                #print("Current Piece ID:", current_piece_id)
                # Use featurizer to convert board state to state index
                state_idx, flipped_state_idx = featurizer(board, piece_id=current_piece_id, move_number=move_number, n_bins=n_bins)
                #print("State index:", state_idx, "Flipped state index:", flipped_state_idx)
                # Ensure state_idx is an integer and within bounds
                if (state_idx > n_bins):
                    state_idx = state_idx % n_bins
                
                # print("-" * 40)
                # print("State index:", state_idx)
                # print(f"Moves in episode: {moves_in_episode}")
                
                # Force drop action after move_before_drop moves
                if moves_in_episode >= move_before_drop:
                    # Force fast drop action    
                    action_val = ACTION_FAST_DROP
                    moves_in_episode = 0  # Reset counter after drop
                else:
                    # Epsilon-greedy action selection
                    if np.random.random() < epsilon:
                        # Exploration: select a random action
                        action_idx = np.random.randint(0, len(possible_actions))
                        # print(f"Exploring: selected random action {action_idx}")
                    else:
                        # Exploitation: select the action with the highest value in the policy matrix
                        action_values = policy_matrix[state_idx, :]
                        action_idx = np.argmax(action_values) # 
                        # print(f"Exploiting: selected best action {action_idx} with value {action_values[action_idx]:.4f}")
                    
                    #would not be necissary if the indexes maped to spot in matrix but 0 is not a response
                    action_val = possible_actions[action_idx]
                    moves_in_episode += 1  # Increment move counter
                    
                    # If flipped_state_idx is True and action is left/right, reverse the action
                    # but keep the original action_idx for policy matrix updates
                    if flipped_state_idx and (action_val == ACTION_MOVE_LEFT or action_val == ACTION_MOVE_RIGHT):
                        # Reverse the action: left -> right, right -> left
                        if action_val == ACTION_MOVE_LEFT:
                            action_val = ACTION_MOVE_RIGHT
                        elif action_val == ACTION_MOVE_RIGHT:
                            action_val = ACTION_MOVE_LEFT
                
                # Store the state-action pair before taking the action (using original action_idx)
                state_action_history.append((state_idx, action_idx))
                
                action_tuple = ActionTuple(
                    discrete=np.array([[action_val]])
                )
                env.set_actions(behavior_name, action_tuple)

        # Save the policy matrix every 100 episodes
        if (episode + 1) % 100 == 0:
            if save_to_file is not None:
                # Save with episode number in filename
                base_name = save_to_file.replace('.npy', '') if save_to_file.endswith('.npy') else save_to_file
                episode_save_path = f"{base_name}_episode_{episode + 1}.npy"
                save_policy_matrix(policy_matrix, episode_save_path)
            else:
                # If no save_to_file provided, use default name with episode number
                episode_save_path = f"policy_matrix_episode_{episode + 1}.npy"
                save_policy_matrix(policy_matrix, episode_save_path)
    
    # Save the policy matrix if save_to_file is provided (final save)
    if save_to_file is not None:
        save_policy_matrix(policy_matrix, save_to_file)
    
    return policy_matrix


def run_policy_matrix(env, policy_matrix, possible_actions, behavior_name, n_episodes=10, n_bins=1000, featurizer=featurize_board):
    """
    Runs a trained policy matrix greedily in the environment to evaluate its performance.
    Uses pure exploitation (no exploration) to display how well the optimized policy performs.
    Useful for comparing different training methods.
    
    Args:
        env: Unity environment instance
        policy_matrix: Trained policy matrix, shape (n_bins, len(possible_actions))
        possible_actions: List of possible action values
        behavior_name: Name of the behavior in the environment
        n_episodes: Number of episodes to run for evaluation
        n_bins: Number of bins used for state space discretization (must match policy_matrix)
        featurizer: Function to featurize the board state (default: featurize_board)
    
    Returns:
        results: Dictionary containing performance metrics (total_reward, avg_reward, episode_rewards)
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING POLICY MATRIX - Running {n_episodes} episodes")
    print(f"{'='*60}")
    
    episode_rewards = []
    total_steps = 0
    
    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        env.reset()
        terminated = False
        episode_reward = 0
        episode_steps = 0
        currentpiece = -1
        move_number = 0
        
        while not terminated:
            # Get decision + terminal steps from Unity
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if len(decision_steps) > 0:
                obs = decision_steps.obs[0]
                agent_ids = decision_steps.agent_id
                
                board, current_piece_id, reward1, reward2 = parse_observation(obs)
                print("reward:", reward1)
                
                # Track move_number similar to epsilonGreedyAgent
                if currentpiece == -1:
                    currentpiece = current_piece_id
                else:
                    if currentpiece != current_piece_id:
                        currentpiece = current_piece_id
                        move_number = 0
                    elif move_number < 8:
                        move_number += 1
                    else:
                        move_number = -1    #Error
                
                # Featurize the board to get the state index
                state_idx, has_piece = featurizer(board, piece_id=current_piece_id, move_number=move_number, n_bins=n_bins)
                
                # Get greedy action from policy matrix (pure exploitation, no exploration)
                action_values = policy_matrix[state_idx, :]
                action_idx = np.argmax(action_values)
                action_val = possible_actions[action_idx]
                
                # Get reward
                if len(terminal_steps) > 0:
                    reward = terminal_steps.reward[0] if hasattr(terminal_steps.reward, "__getitem__") else terminal_steps.reward
                else:
                    reward = reward1
                
                episode_reward += reward
                episode_steps += 1
                
                # Execute action
                action_tuple = ActionTuple(
                    discrete=np.array([[action_val]])
                )
                env.set_actions(behavior_name, action_tuple)
            
            # Step environment
            env.step()
            
            # Check if episode ended (get steps after the step)
            decision_steps_after, terminal_steps_after = env.get_steps(behavior_name)
            if len(terminal_steps_after) > 0:
                reward = terminal_steps_after.reward
                if hasattr(reward, "__getitem__"):
                    reward = reward[0]
                episode_reward += reward
                print(f"  Episode ended with total reward: {episode_reward:.2f} (steps: {episode_steps})")
                terminated = True
        
        episode_rewards.append(episode_reward)
        total_steps += episode_steps
    
    # Calculate statistics
    total_reward = sum(episode_rewards)
    avg_reward = total_reward / n_episodes
    max_reward = max(episode_rewards)
    min_reward = min(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    # Display results
    print(f"\n{'='*60}")
    print("POLICY MATRIX EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes run: {n_episodes}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per episode: {avg_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Std deviation: {std_reward:.2f}")
    print(f"Average steps per episode: {total_steps / n_episodes:.1f}")
    print(f"\nEpisode rewards: {[f'{r:.2f}' for r in episode_rewards]}")
    print(f"{'='*60}\n")
    
    results = {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards,
        'n_episodes': n_episodes
    }
    
    return results


def parse_observation(obs):
    """
    Given the ML-Agents observation tensor, extract:
      - First 200 values as a 20x10 numpy array (the "board")
      - Next values: current_piece_id, reward1, reward2 (floats)
    Returns:
      board (np.ndarray shape 20x10), current_piece_id (float/int), reward1, reward2 (floats)
    """
    if obs.size == 0:
        return None, None, None, None
    flat_obs = obs.flatten()
    board = flat_obs[:200].reshape((20, 10))
    # Remaining values: assume the order is [piece_id, reward1, reward2]
    if len(flat_obs) >= 203:
        current_piece_id = flat_obs[200]
        reward1 = flat_obs[201]
        reward2 = flat_obs[202]

    else:
        current_piece_id, reward1, reward2 = None, None, None
    return board, current_piece_id, reward1, reward2







# TODO: create a simple function that steps through the environment recives the returns from environment and prints them out
def stepThroughEnvironment(env,possible_actions):
    decision_steps, terminal_steps = env.get_steps(possible_actions)
    for agent_id in decision_steps:
        if len(decision_steps) > 0:
            obs = decision_steps.obs[0]
            agent_ids = decision_steps.agent_id
            action_val = np.random.randint(0, len(possible_actions))
            action_tuple = ActionTuple(
                discrete=np.array([[action_val]])
            )

    env.step()
    return decision_steps, terminal_steps


def updatePolicyMatrix(policy_matrix, state, action, reward):
    policy_matrix[state, action] += reward
    return policy_matrix    