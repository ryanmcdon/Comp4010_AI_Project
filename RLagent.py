from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

env = UnityEnvironment(file_name=None)  # Connect to Unity
env.reset()

behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

print("Connected to:", behavior_name)

for episode in range(2000):
    env.reset()
    terminated = False

    while not terminated:
        # Get current decision requests
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # For agents needing a decision:
        if len(decision_steps) > 0:
            # OBSERVATIONS
            obs = decision_steps.obs[0]  # first (and only) observation tensor

            # ----- YOUR RL ALGORITHM GOES HERE -----
            # Example: Random policy
            action = np.random.randint(0, spec.action_spec.discrete_branches[0], size=(len(decision_steps), 1))
            # ---------------------------------------

            action_tuple = ActionTuple(discrete=action)
            env.set_actions(behavior_name, action_tuple)

        # Apply step
        env.step()

        # Check if episode ended
        if len(terminal_steps) > 0:
            terminated = True

env.close()