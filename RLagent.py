from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import time
from featurizer import featurize_board

#TODO create a policy matrix randomly that is updated with the reward from randomAgent

# Define possible actions and corresponding integer mappings
# For example, suppose the game allows 6 actions: move left, move right, rotate, drop, hold, and soft drop.
# You can modify or expand this list according to your environment's action space.
ACTION_MOVE_LEFT = 1
ACTION_MOVE_RIGHT = 2
ACTION_ROTATE = 3
ACTION_DROP = 4


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
                print("Board:", board)
                print("Current Piece ID:", current_piece_id)
                # Use new featurizer instead of old contouring functions
                state_idx = featurize_board(board, piece_id=current_piece_id, n_bins=n_bins)
                print("State index:", state_idx)
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


# An epsilon-greedy agent that uses the policy matrix to select actions
def epsilonGreedyAgent(env, possible_actions, behavior_name, gamma=0.99, epsilon=0.1, max_episodes=400, n_bins=1000):
    """
    Epsilon-greedy agent that selects actions based on the policy matrix.
    With probability epsilon, selects a random action (exploration).
    With probability (1-epsilon), selects the best action according to the policy matrix (exploitation).
    
    Returns:
        policy_matrix: The optimized policy matrix after training, shape (n_bins, len(possible_actions))
    """
    # Policy matrix initialization: shape (n_bins, actionspace=len(possible_actions))
    # Using featurize_board to map board states to n_bins discrete states
    # Using float32 to reduce memory usage (half the size of float64)
    policy_matrix = np.random.rand(n_bins, len(possible_actions)).astype(np.float32)
    print("Policy matrix shape:", policy_matrix.shape)
    print(f"Using epsilon-greedy policy with epsilon={epsilon}")
    
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
                print("Board:", board)
                print("Current Piece ID:", current_piece_id)
                # Use new featurizer instead of old contouring functions
                state_idx = featurize_board(board, piece_id=current_piece_id, n_bins=n_bins)
                
                print("-" * 40)
                print("State index:", state_idx)
  
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    # Exploration: select a random action
                    action_idx = np.random.randint(0, len(possible_actions))
                    print(f"Exploring: selected random action {action_idx}")
                else:
                    # Exploitation: select the action with the highest value in the policy matrix
                    action_values = policy_matrix[state_idx, :]
                    action_idx = np.argmax(action_values) # +1 because the action_values are 0-indexed but the possible_actions are 1-indexed
                    print(f"Exploiting: selected best action {action_idx} with value {action_values[action_idx]:.4f}")
                
                action_val = possible_actions[action_idx]
                    
                # Update the policy matrix based on the reward received for the action
                
                # This is similar to a value update but not a full RL algorithm
                if len(terminal_steps) > 0:
                    # Use the true reward at the end of the episode
                    reward = terminal_steps.reward[0] if hasattr(terminal_steps.reward, "__getitem__") else terminal_steps.reward
                else:
                    # Use intermediate reward if available
                    reward = reward1


                # Basic update rule: nudges the value estimate for this state-action pair toward the received reward
                # Learning rate alpha is set arbitrarily small
                alpha = 0.1
                policy_matrix[state_idx, action_idx] = policy_matrix[state_idx, action_idx] + alpha * (reward - policy_matrix[state_idx, action_idx])

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
    return policy_matrix


def run_policy_matrix(env, policy_matrix, possible_actions, behavior_name, n_episodes=10, n_bins=1000):
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
        
        while not terminated:
            # Get decision + terminal steps from Unity
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if len(decision_steps) > 0:
                obs = decision_steps.obs[0]
                agent_ids = decision_steps.agent_id
                
                board, current_piece_id, reward1, reward2 = parse_observation(obs)
                
                # Featurize the board to get the state index
                state_idx = featurize_board(board, piece_id=current_piece_id, n_bins=n_bins)
                
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
            
            # Check if episode ended
            if len(terminal_steps) > 0:
                reward = terminal_steps.reward
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






# -------------------------------------------------
# CONNECT TO UNITY
# -------------------------------------------------
env = UnityEnvironment(file_name=None)  # Connects to Editor Play mode
env.reset()

# Get the single behavior
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

print("Connected to:", behavior_name)
print("Observation Specs:", spec.observation_specs)
print("Action Spec:", spec.action_spec)
print("-" * 60)
print("Possible actions:", behavior_name)
# Number of discrete actions (should be 6 for your TetrisAgent)
n_actions = spec.action_spec.discrete_branches[0]

policy_matrix = epsilonGreedyAgent(env,possible_actions,behavior_name) # Call the epsilonGreedyAgent function
print("Policy matrix: test run")
run_policy_matrix(env,policy_matrix,possible_actions,behavior_name) # Call the run_policy_matrix function
#randomAgent(env,possible_actions,behavior_name) # Call the randomAgent function


# Close Unity
env.close()



# TODO: create a simple function that steps through the environment recives the returns from environment and prints them out
def stepThroughEnvironment(env,possible_actions):
    decision_steps, terminal_steps = env.get_steps(possible_actions)
    for agent_id in decision_steps:
        if len(decision_steps) > 0:
            obs = decision_steps.obs[0]
            agent_ids = decision_steps.agent_id
            action_val = np.random.randint(0, n_actions)
            action_tuple = ActionTuple(
                discrete=np.array([[action_val]])
            )

    env.step()
    return decision_steps, terminal_steps


def updatePolicyMatrix(policy_matrix, state, action, reward):
    policy_matrix[state, action] += reward
    return policy_matrix    