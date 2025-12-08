import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import os
import glob

class DuelingQNetwork(torch.nn.Module):
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.value_stream = torch.nn.Linear(256, 1)
        self.adv_stream = torch.nn.Linear(256, action_size)
        self.act = torch.nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        adv_mean = adv.mean(dim=1, keepdim=True)
        return value + adv - adv_mean



def load_latest_checkpoint(policy_net, device):
    checkpoint_files = glob.glob("checkpoints/*.pth")

    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in /checkpoints!")

    latest = max(checkpoint_files, key=os.path.getctime)
    print(f"\n🔄 Loading checkpoint: {latest}")

    checkpoint = torch.load(latest, map_location=device)
    policy_net.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from episode: {checkpoint['episode']}")

    return checkpoint



def flatten_state(obs):
    arr = np.array(obs, dtype=np.float32)
    return arr.reshape(-1)



if __name__ == "__main__":

    state_size = 28
    possible_actions = [0, 1, 2, 3, 4]
    n_actions = len(possible_actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    env = UnityEnvironment(
        file_name="DQN-Unity/Tetris",
        no_graphics=False,
        seed=1,
        additional_args=[
            "-screen-fullscreen", "0",
            "-screen-width", "800",
            "-screen-height", "800"
        ]
    )

    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"Connected to {behavior_name}")

    policy_net = DuelingQNetwork(state_size, n_actions).to(device)

    checkpoint = load_latest_checkpoint(policy_net, device)

    eval_episodes = 10

    for ep in range(eval_episodes):

        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)
        state = flatten_state(decision_steps.obs[0])

        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action_index = int(policy_net(state_tensor).argmax())

            action_value = possible_actions[action_index]
            action_tuple = ActionTuple(discrete=np.array([[action_value]], dtype=np.int32))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                reward = float(terminal_steps.reward[0])
                total_reward += reward
                done = True
            elif len(decision_steps) > 0:
                reward = float(decision_steps.reward[0])
                total_reward += reward
                state = flatten_state(decision_steps.obs[0])

        print(f"Episode {ep+1}/{eval_episodes} | Reward = {total_reward:.2f}")

    env.close()

