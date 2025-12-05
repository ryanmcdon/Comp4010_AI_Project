import numpy as np
import os

class LinearQAgent:
    def __init__(self,
                 n_actions,
                 n_features,
                 alpha=0.01,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.999):

        self.n_actions = n_actions
        self.n_features = n_features

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay


        self.W = 0.01 * np.random.randn(n_features, n_actions).astype(np.float32)

    def q_values(self, phi):
        return phi @ self.W  

    def select_action(self, phi):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values(phi)))

    def learn(self, phi, action, reward, next_phi, done):
        q_val = self.q_values(phi)[action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_values(next_phi))

        td_error = target - q_val


        self.W[:, action] += self.alpha * td_error * phi

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return td_error

    def save(self, path="models/W.npy"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.W)

    def load(self, path="models/W.npy"):
        self.W = np.load(path)
