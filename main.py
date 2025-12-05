from mlagents_envs.environment import UnityEnvironment
from RLagent import epsilonGreedyAgent, run_policy_matrix, possible_actions
from featurizer import featurize_board_4x4, featurize_board_4x4_no_center
from AI_learning import QLearningConfig, train_and_optionally_eval
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
# Number of discrete actions (should be 6 for TetrisAgent)
n_actions = spec.action_spec.discrete_branches[0]

# Example usage with save/load functionality:
# To load an existing policy matrix: policy_matrix = epsilonGreedyAgent(env, possible_actions, behavior_name, load_from_file='policy_matrix.npy')
# To save after training: policy_matrix = epsilonGreedyAgent(env, possible_actions, behavior_name, save_to_file='policy_matrix.npy')
# To both load and save: policy_matrix = epsilonGreedyAgent(env, possible_actions, behavior_name, load_from_file='policy_matrix.npy', save_to_file='policy_matrix.npy')

#policy_matrix = epsilonGreedyAgent(env, possible_actions, behavior_name, max_episodes=8000, epsilon=0.01, move_before_drop=40, load_from_file='policy_matrix_4x4_1500.npy', save_to_file='policy_matrix_4x4.npy', featurizer=featurize_board_4x4_no_center,n_bins=24000) # Call the epsilonGreedyAgent function and save the matrix
#print("Policy matrix: test run")


cfg = QLearningConfig(episodes=800, save_path="q_table.pkl")
q_table = train_and_optionally_eval(env, behavior_name, config=cfg, eval_episodes=10)
#run_policy_matrix(env, policy_matrix, possible_actions, behavior_name, featurizer=featurize_board_4x4, n_bins=10000) # Call the run_policy_matrix function with matching featurizer and n_bins
#randomAgent(env,possible_actions,behavior_name) # Call the randomAgent function


# Close Unity
env.close()

