from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

env = UnityEnvironment(file_name=None)  # Connect to Unity
env.reset()

behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

print("Connected to:", behavior_name)

try:
    for episode in range(10):
        env.reset()

        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # If the agent ended last frame
            if len(terminal_steps) > 0:
                print("Episode ended")
                break

            # Make sure we have data to act on
            if len(decision_steps) > 0:
                '''
                action = np.random.randint(
                    spec.action_spec.discrete_branches[0],
                    size=(len(decision_steps), 1)'''
                branch_size = spec.action_spec.discrete_branches[0]
                
                actions = np.random.randint(branch_size, size=(len(decision_steps), 1))
                env.set_actions(behavior_name, ActionTuple(discrete=actions))

            env.step()

    # Check if episode ended
    if len(terminal_steps) > 0:
        terminated = True

finally:
    env.close()