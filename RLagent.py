from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import time
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

# Number of discrete actions (should be 6 for your TetrisAgent)
n_actions = spec.action_spec.discrete_branches[0]

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
for episode in range(2000):
    print(f"\n========= EPISODE {episode} START =========")
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

            # Print first 10 obs values for debug
            #print("[OBS] first 10 values:", obs.flatten()[:10])

            # Random action (0..n_actions-1)
            #time.sleep(0.1)  # slight delay for readability
            action_val = np.random.randint(0, n_actions)
            #print("[ACT] sending action:", action_val)

            # Send action in ML-Agents tuple format
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

# Close Unity
env.close()
