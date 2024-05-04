import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, alpha):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_actions)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x): 
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Critic(nn.Module):
    def __init__(self, input_dims, alpha):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(*self.input_dims, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():
    def __init__(self, input_dims, n_actions, alpha, beta, gamma=0.99):
        super(Agent, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.actor = Actor(self.input_dims, self.n_actions, alpha=alpha)
        self.critic = Critic(self.input_dims, alpha=beta)
        self.gamma = gamma
        self.log_probs = None
        
    def choose_action(self, observation):
        mu, sigma = self.actor.forward(observation)
        sigma = torch.exp(sigma)
        action_probs = torch.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=torch.Size([1]))
        self.log_probs = action_probs.log_prob(probs)
        action = torch.tanh(probs)

        return action.item()
        
    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        value_1 = self.critic.forward(state)
        # print(value_1)
        value_2 = self.critic.forward(new_state)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)

        delta = (reward + self.gamma * value_2 * (1-int(done))) - value_1

        critic_loss = delta ** 2
        # print(critic_loss)
        actor_loss = -self.log_probs * delta

        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()