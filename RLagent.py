import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import matplotlib.pyplot as plt
import json
import os


class DuelingQNetwork(nn.Module):
    def __init__(self, input_size: int, action_size: int):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.value_stream = nn.Linear(256, 1)
        self.adv_stream = nn.Linear(256, action_size)

        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        value = self.value_stream(x)
        adv = self.adv_stream(x)

        adv_mean = adv.mean(dim=1, keepdim=True)
        return value + adv - adv_mean


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        prios = self.priorities[: len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, new_priorities):
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = prio


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = PrioritizedReplayBuffer(capacity=100_000)
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9997

        self.batch_size = 256
        self.lr = 1e-4

        self.beta = 0.4
        self.beta_increment = 1e-5
        self.per_eps = 1e-5

        self.target_update_freq = 1000
        self.train_steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")

        self.policy_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.writer = SummaryWriter("runs/tetris_advanced_dqn")

        self.last_loss = None

    def remember(self, s, a, r, ns, d):
        self.memory.push((s, a, r, ns, d))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state).argmax())

    def replay(self):
        if len(self.memory) < max(5000, self.batch_size):
            return

        batch, indices, weights = self.memory.sample(self.batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones       = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        weights = weights.to(self.device).unsqueeze(1)

        curr_Q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_Q = self.target_net(next_states).gather(1, next_actions)
            target_Q = rewards + self.gamma * next_Q * (1 - dones)

        td_errors = (curr_Q - target_Q).detach().abs().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors + self.per_eps)

        loss = (self.loss_fn(curr_Q, target_Q) * weights).mean()
        self.last_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


possible_actions = [0, 1, 2, 3, 4]
n_actions = len(possible_actions)

state_size = 28

reward_history = []
epsilon_history = []
loss_history = []


def flatten_state(obs):
    arr = np.array(obs, dtype=np.float32)
    return arr.reshape(-1)


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

agent = DQNAgent(state_size, n_actions)

episodes = 15000

os.makedirs("checkpoints", exist_ok=True)

for e in range(episodes):

    env.reset()
    decision_steps, _ = env.get_steps(behavior_name)
    state = flatten_state(decision_steps.obs[0])

    total_reward = 0
    done = False
    steps = 0

    while not done:
        action_index = agent.act(state)
        action_value = possible_actions[action_index]

        action_tuple = ActionTuple(discrete=np.array([[action_value]], dtype=np.int32))
        env.set_actions(behavior_name, action_tuple)
        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(terminal_steps) > 0:
            reward = float(terminal_steps.reward[0])
            agent.remember(state, action_index, reward, np.zeros(state_size), True)
            total_reward += reward
            done = True

        elif len(decision_steps) > 0:
            reward = float(decision_steps.reward[0])

            next_state = flatten_state(decision_steps.obs[0])
            agent.remember(state, action_index, reward, next_state, False)

            state = next_state
            total_reward += reward
            steps += 1

        agent.replay()

    reward_history.append(total_reward)
    epsilon_history.append(agent.epsilon)
    loss_history.append(agent.last_loss)

    if e % 500 == 0 and e > 0:
        torch.save({
            "episode": e,
            "model_state": agent.policy_net.state_dict(),
            "target_state": agent.target_net.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon
        }, f"checkpoints/ep_{e}.pth")
        print(f"saved checkpoint at ep {e}")

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"Ep {e}/{episodes} | Reward={total_reward:.2f} | Eps={agent.epsilon:.3f}")

env.close()

os.makedirs("training_logs", exist_ok=True)

with open("training_logs/history.json", "w") as f:
    json.dump({
        "reward": reward_history,
        "epsilon": epsilon_history,
        "loss": loss_history
    }, f)

plt.figure()
plt.plot(reward_history)
plt.title("Reward")
plt.savefig("training_logs/reward.png")

plt.figure()
plt.plot(epsilon_history)
plt.title("Epsilon")
plt.savefig("training_logs/epsilon.png")

plt.figure()
plt.plot([x for x in loss_history if x])
plt.title("Loss")
plt.savefig("training_logs/loss.png")
