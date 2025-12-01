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
    for episode in range(2000):
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


# OLD CONTOURING FUNCTIONS - COMMENTED OUT, USING featurize_board INSTEAD
# def board_contour(board):
#     """
#     Given a 20x10 numpy array (values 0 or 1), return the 'contour':
#     A length-10 array where the i-th entry is the lowest (largest row index)
#     such that board[row, i] == 1, or -1 if none exists in that column.
#     The contour is measured from the top (row 0) to bottom (row 19).
#     """
#     board = np.array(board)
#     rows, cols = board.shape
#     contour = np.full(cols-1, -1, dtype=int)
#     prev = 0
#     for col in range(cols):
#         ones_idx = np.where(board[:, col] != 0)[0]
#         diff = 0
#         if len(ones_idx) > 0:
#             # Calculate difference with previous column's height and bound to [-3, +2]
#             if col != 0:
#                 if prev == -1 or len(ones_idx) == 0:
#                     diff = 0
#                 else:
#                     diff = prev - ones_idx.max()
#                     diff = max(-3, min(2, diff))
#             contour[col-1] = diff
#             prev = ones_idx.max()
# #            contour[col] = ones_idx.max()
#     return contour


# def _contour_to_state_number(contour):
#     """
#     Helper function: Convert a contour array to a state number without canonicalization.
#     """
#     base = 6
#     max_len = 9

#     # Truncate or pad contour to length 9
#     contour = np.array(contour)
#     if contour.shape[0] > max_len:
#         contour_short = contour[:max_len]
#     elif contour.shape[0] < max_len:
#         # pad with zeros
#         contour_short = np.pad(contour, (0, max_len - contour.shape[0]), 'constant')
#     else:
#         contour_short = contour

#     # Map from [-3, +2] -> [0, 5] (base-6 encoding)
#     mapped = np.clip(contour_short, -3, 2) + 3

#     # Convert to state number (base-6 number)
#     state_number = 0
#     for i, val in enumerate(mapped):
#         state_number += int(val) * (base ** i)

#     return state_number


# def contour_to_state(contour):
#     """
#     Given a 9-element (9x1) contour array, compute a canonical state number.
#     Mirrored states are mapped to the same canonical state (the lexicographically smaller one).
#     This reduces the state space by approximately half.
#     
#     Returns:
#         tuple: (canonical_state_number, is_mirrored)
#             - canonical_state_number: The canonical state number (0..~6^9/2)
#             - is_mirrored: Boolean indicating if the mirror was used to get the canonical state
#     """
#     contour = np.array(contour)
#     
#     # Compute state number for original contour
#     state_original = _contour_to_state_number(contour)
#     
#     # Compute state number for mirrored contour (reversed)
#     contour_mirrored = np.flip(contour)
#     state_mirrored = _contour_to_state_number(contour_mirrored)
#     
#     # Use the smaller state number as canonical (reduces state space by ~half)
#     if state_mirrored < state_original:
#         return state_mirrored, True
#     else:
#         return state_original, False


# #returns the action mirrored (for when the state was mirrored)
# def mirror_action(action):
#     canonical_actions = [0, 1, 2, 3, 4, 5]
#     mirrored_actions = [0, 2, 1, 3, 4, 5]
#     return mirrored_actions[action]







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

randomAgent(env,possible_actions,behavior_name) # Call the randomAgent function

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