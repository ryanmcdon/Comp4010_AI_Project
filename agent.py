import time
import numpy as np
from env import env

env = env()
state, _ = env.reset()

for step in range(200):  
    action = env.action_space.sample()  
    state, reward, done, _, _ = env.step(action)
    print(f"Step {step}: took action {action}, reward {reward}")
    time.sleep(0.01)
