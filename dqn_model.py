import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class TetrisNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(TetrisNetwork, self).__init__()
        

        self.fc1 = nn.Linear(28, 128)
        

        self.fc2 = nn.Linear(128, 128)
        

        self.fc3 = nn.Linear(128, 64)
        

        self.output = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.output(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        

        self.memory = deque(maxlen=20000) 
        self.gamma = 0.99      
        self.epsilon = 1.0     
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.batch_size = 64


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")


        self.model = TetrisNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        

        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([[i[1]] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([[i[2]] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([[i[4]] for i in minibatch])).to(self.device)

        curr_Q = self.model(states).gather(1, actions)
        

        next_Q = self.model(next_states).max(1)[0].unsqueeze(1)
        expected_Q = rewards + (self.gamma * next_Q * (1 - dones))

        loss = self.loss_fn(curr_Q, expected_Q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay