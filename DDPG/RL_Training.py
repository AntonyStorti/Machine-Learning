import torch.nn.functional as F
from collections import deque
import torch.optim as optim
import torch.nn as nn
from math import pi
import numpy as np
import random
import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(CriticNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.buffer):
            raise IndexError("Indice fuori dal range del buffer.")
        return self.buffer[idx]

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def update_model(actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer, replay_buffer, batch_size, gamma, tau):

    # Prelevo casualmente 4 array numpy dal ReplayBuffer --> Sono combinazioni casuali dei samples presenti
    states, actions, rewards, next_states = replay_buffer.sample(batch_size)

    # Li converto in tensori
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    # Critic update
    next_actions = target_actor(next_states)
    next_q_values = target_critic(next_states, next_actions)
    target_q_values = rewards + gamma * next_q_values
    q_values = critic_model(states, actions)
    critic_loss = F.mse_loss(q_values, target_q_values.detach())

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor update
    policy_loss = -critic_model(states, actor_model(states)).mean()

    actor_optimizer.zero_grad()
    policy_loss.backward()
    actor_optimizer.step()

    # Soft update of target networks
    for target_param, param in zip(target_critic.parameters(), critic_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(target_actor.parameters(), actor_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def start_RL_Training():

    # Addestro la Rete DDPG:
    hidden_size = 64
    input_size = 3
    output_size = 2
    noise = OUNoise(output_size)

    actor_model = ActorNetwork(input_size, hidden_size, output_size)
    critic_model = CriticNetwork(input_size, hidden_size, output_size)
    target_actor = ActorNetwork(input_size, hidden_size, output_size)
    target_critic = CriticNetwork(input_size, hidden_size, output_size)

    # Carico i modelli salvati, se esistono:
    if os.path.exists('DDPG/actor_model.pth') and os.path.exists('DDPG/critic_model.pth'):
        actor_model.load_state_dict(torch.load('DDPG/actor_model.pth'))
        critic_model.load_state_dict(torch.load('DDPG/critic_model.pth'))
        target_actor.load_state_dict(torch.load('DDPG/actor_model.pth'))
        target_critic.load_state_dict(torch.load('DDPG/critic_model.pth'))
        print("Modelli caricati dai file salvati.")
    else:
        # Inizializzazione delle reti target
        target_actor.load_state_dict(actor_model.state_dict())
        target_critic.load_state_dict(critic_model.state_dict())
        print("Modelli non trovati, si parte da zero.")

    actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.001)

    return noise, actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer


def main(state, cli, noise, actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer, replay_buffer):

    epsilon = 1.0
    epsilon_decay = 0.995
    batch_size = 64
    gamma = 0.99
    tau = 0.001

    noise.reset()

    state_tensor = torch.tensor(state[17:20], dtype=torch.float32).unsqueeze(0)

    # Gestisco gli ultimi due giunti (quelli del Paddle)
    # MOTIVO --> Sono gli unici che vengono gestiti dal RL
    action = actor_model(state_tensor).detach().numpy().flatten() + noise.noise()
    action[0] = np.clip(action[0], -pi * 3 / 4, pi * 3 / 4)  # Clip delle azioni del Giunto 9 --> +- 45°

    # Invio dei giunti --> (Da 0 a 8): MPL | (Da 9 a 10): DDPG
    joints = [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], action[0], action[1]]
    cli.send_joints(joints)

    # Attendo che la pallina colpisca il tavolo avversario, oppure, termini la giocata corrente
    # MOTIVO --> Assegnazione corretta del reward della giocata corrente
    while True:

        next_state = cli.get_state()
        if next_state[33] == 1 or next_state[26] == 1 or next_state[27] == 1:
            break

    # Assegno il reward se la pallina che ho lanciato ha colpito il semi-campo avversario
    if next_state[33] == 1:
        reward = 1
    else:
        reward = -1

    # Aggiungo nel ReplayBuffer solo le esperienze con reward positivo
    if reward == 1:
        replay_buffer.add(state[17:20], action, reward, next_state[17:20])

    # Se il ReplayBuffer è oltre i limiti del MiniBatch --> Salvo i modelli + Svuoto il buffer
    if replay_buffer.size() >= batch_size*2:

        update_model(actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer, replay_buffer, batch_size, gamma, tau)

        torch.save(actor_model.state_dict(), 'actor_model.pth')
        torch.save(critic_model.state_dict(), 'critic_model.pth')
        replay_buffer.clear()

        print("Modelli salvati.")

    # Decay epsilon
    epsilon *= epsilon_decay

