import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class policyNet(nn.Module):
    def __init__(self, input_dim, n_action, alpha):
        super(policyNet, self).__init__()
        self.input_dim = input_dim
        self.n_action = n_action
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_action)
        self.relu = nn.ReLU()

        self.optim = optim.Adam(self.parameters(), alpha)
    
    def forward(self, x):
        x = torch.tensor(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        probs = F.softmax(self.fc3(x))

        return probs

class Agent():
    def __init__(self, input_dim, n_action, alpha, gamma=0.99):
        self.input_dim = input_dim
        self.n_action = n_action
        self.policy = policyNet(self.input_dim, self.n_action, alpha)
        self.gamma = gamma
        self.log_probs = None
        self.reward_memory = []
        self.action_memory = []

    def choose_action(self, observation):
        probs = self.policy.forward(observation)
        # print(probs)
        action_probs = torch.distributions.Categorical(probs)
        # print(action_probs)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        self.action_memory.append(self.log_probs)
        return action.item()
    
    def store_reward(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        total_loss = 0
        self.policy.optim.zero_grad()
        for i in range(len(self.action_memory)):
            G = 0
            discount = 1
            for j in range(i, len(self.action_memory)):
                G += self.reward_memory[j] * discount
                discount *= self.gamma
            G = torch.tensor(G, requires_grad=True)
            loss = -G * self.action_memory[i]
            total_loss += loss
        print(total_loss)
        total_loss.backward()
        self.policy.optim.step()
        self.reward_memory = []
        self.action_memory = []
    
    def learn_2(self):
        self.policy.optim.zero_grad()
        # Assumes only a single episode for reward_memory
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        # G = (G - mean) / std

        G = torch.tensor(G, dtype=torch.float)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        print(loss)
        loss.backward()
        self.policy.optim.step()

        self.action_memory = []
        self.reward_memory = []
