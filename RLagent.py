from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from dqn_model import DQNAgent


possible_actions = [0, 1, 2, 3, 4, 5]
n_actions = len(possible_actions)


state_size = 28 

def parse_state(obs):

    return obs.flatten()


if __name__ == "__main__":

    env = UnityEnvironment(file_name=None)
    env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"Connected to {behavior_name}")
    print("Starting Deep Q-Learning Training...")


    agent = DQNAgent(state_size, n_actions)
    
    episodes = 2000
    
    for e in range(episodes):
        env.reset()
        

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(decision_steps) == 0: continue 
        
        obs = decision_steps.obs[0]
        state = parse_state(obs)
        
        total_reward = 0
        done = False
        
        while not done:

            action_idx = agent.act(state)
            action_val = possible_actions[action_idx]
            

            action_tuple = ActionTuple(discrete=np.array([[action_val]], dtype=np.int32))
            env.set_actions(behavior_name, action_tuple)
            env.step()
            

            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if len(terminal_steps) > 0:

                reward = float(terminal_steps.reward[0])
                total_reward += reward
                

                next_state = np.zeros(state_size)
                

                agent.remember(state, action_idx, reward, next_state, True)
                done = True
                
            elif len(decision_steps) > 0:

                reward = float(decision_steps.reward[0])
                total_reward += reward
                
                next_obs = decision_steps.obs[0]
                next_state = parse_state(next_obs)
                

                agent.remember(state, action_idx, reward, next_state, False)
                state = next_state
            

            agent.replay()
            
        print(f"Ep {e}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f} | Mem: {len(agent.memory)}")

    env.close()