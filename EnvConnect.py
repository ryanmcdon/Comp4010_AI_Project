from mlagents_envs.environment import UnityEnvironment
import numpy as np

# Launch the Unity environment (Editor or built build)
env = UnityEnvironment(file_name=None)  # None = connect to Editor

env.reset()

# Get behavior name
behavior_name = list(env.behavior_specs)[0]
print("Behavior:", behavior_name)

spec = env.behavior_specs[behavior_name]
print("Observation size:", spec.observation_specs[0].shape)
print("Action type:", spec.action_spec)