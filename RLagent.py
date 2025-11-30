from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import time
from TetrisGreedy import greedy_policy
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
    episode_return = 0.0

    while not terminated:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # --- take an action if agent is requesting one ---
        if len(decision_steps) > 0:
            agent_ids = decision_steps.agent_id
            obs = np.concatenate([o.flatten() for o in decision_steps.obs])  # assuming single observation

            action_val = greedy_policy(obs)
            print("ACTION SELECTED:", action_val)


            action_tuple = ActionTuple(
                discrete=np.array([[action_val]])
            )
            env.set_actions(behavior_name, action_tuple)

            # reward for this non-terminal step (if any)
            # (usually per-step reward is on decision_steps)
            episode_return += float(decision_steps.reward[0])
            
            print("OBS:", obs.flatten()[:10])

        # step environment
        env.step()

        # check terminal
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(terminal_steps) > 0:
            # add reward from the final step too
            episode_return += float(terminal_steps.reward[0])
            print("[RET]  Episode cumulative reward:", episode_return)
            terminated = True

env.close()
