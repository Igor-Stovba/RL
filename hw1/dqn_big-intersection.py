import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
from stable_baselines3.dqn.dqn import DQN

from sumo_rl import SumoEnvironment


env = SumoEnvironment(
    net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
    single_agent=True,
    route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
    out_csv_name="outputs/big-intersection/dqn",
    use_gui=True,
    num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
)


class DoubleDQN:
    """Double DQN agent."""

    def __init__(
        self,
        env,
        policy="MlpPolicy",
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=50000,
        batch_size=32,
        tau=1.0,
        target_update_interval=500,
        train_freq=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        verbose=1,
    ):
        self.env = env
        self.action_space = env.action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval

        self.eps = exploration_initial_eps
        self.eps_decay = (exploration_initial_eps - exploration_final_eps) / (
            exploration_fraction
        )
        self.eps_min = exploration_final_eps

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # Q-networks
        self.q = MLP(obs_dim, act_dim)
        self.q_target = MLP(obs_dim, act_dim)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size, obs_dim)

        self.step_count = 0

    def predict(self, obs, deterministic=False):
        if not deterministic and np.random.rand() < self.eps:
            return self.action_space.sample()

        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q(obs)
        return torch.argmax(q_vals, dim=1).item()

    def learn(self, total_timesteps):
        obs, _ = self.env.reset()
        for t in range(total_timesteps):
            action = self.predict(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            done = terminated or truncated
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            # Exploration decay
            if self.eps > self.eps_min:
                self.eps -= self.eps_decay

            # Train
            if t > self.batch_size and t % self.train_freq == 0:
                self._train_step()

            # Update target net
            if t % self.target_update_interval == 0:
                self.q_target.load_state_dict(self.q.state_dict())

            if done:
                obs, _ = self.env.reset()

    def _train_step(self):
        obs, act, rew, next_obs, done = self.replay_buffer.sample(self.batch_size)

        obs = torch.FloatTensor(obs)
        act = torch.LongTensor(act)
        rew = torch.FloatTensor(rew)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done)

        with torch.no_grad():
            next_q_values = self.q(next_obs)
            next_actions = torch.argmax(next_q_values, dim=1)

            next_q_target_values = self.q_target(next_obs)
            target = rew + self.gamma * (1 - done) * next_q_target_values[
                torch.arange(self.batch_size), next_actions
            ]

        q_vals = self.q(obs)[torch.arange(self.batch_size), act]
        loss = torch.nn.functional.mse_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, size, obs_dim):
        self.size = size
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((size, obs_dim))
        self.next_obs = np.zeros((size, obs_dim))
        self.actions = np.zeros(size, dtype=int)
        self.rewards = np.zeros(size)
        self.dones = np.zeros(size)

    def add(self, obs, action, reward, next_obs, done):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = done


        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_idx, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out),
        )

    def forward(self, x):
        return self.net(x)



class PrioritizedReplayBuffer:
    def __init__(self, size, obs_dim, alpha=0.6, beta_start=0.4, beta_frames=50000):
        self.size = size
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((size, obs_dim))
        self.next_obs = np.zeros((size, obs_dim))
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size)
        self.dones = np.zeros(size, dtype=np.float32)

        self.priorities = np.zeros(size, dtype=np.float32)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def add(self, obs, action, reward, next_obs, done):
        idx = self.ptr

        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = done

        max_prio = self.priorities.max() if self.ptr > 0 else 1.0
        self.priorities[idx] = max_prio

        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def _beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def sample(self, batch_size):
        self.frame += 1
        beta = self._beta()

        max_idx = self.size if self.full else self.ptr
        prios = self.priorities[:max_idx] ** self.alpha
        probs = prios / prios.sum()

        idx = np.random.choice(max_idx, batch_size, p=probs)

        weights = (max_idx * probs[idx]) ** (-beta)
        weights /= weights.max()

        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            idx,
            weights.astype(np.float32),
        )

    def update_priorities(self, idx, priorities):
        self.priorities[idx] = priorities


class MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out),
        )

    def forward(self, x):
        return self.net(x)


class PrioritizedDQN:

    def __init__(
        self,
        env,
        policy="MlpPolicy",
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=50000,
        batch_size=32,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        verbose=1,
    ):
        self.env = env
        self.action_space = env.action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval

        self.eps = exploration_initial_eps
        self.eps_decay = (exploration_initial_eps - exploration_final_eps) / (
            exploration_fraction
        )
        self.eps_min = exploration_final_eps

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # Q networks
        self.q = MLP(obs_dim, act_dim)
        self.q_target = MLP(obs_dim, act_dim)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)


        self.replay = PrioritizedReplayBuffer(buffer_size, obs_dim)

        self.step_count = 0

    def predict(self, obs, deterministic=False):
        if not deterministic and np.random.rand() < self.eps:
            return self.action_space.sample()

        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(obs)
        return torch.argmax(qvals).item()

    def learn(self, total_timesteps):
        obs, _ = self.env.reset()

        for t in range(total_timesteps):
            action = self.predict(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay.add(obs, action, reward, next_obs, done)
            obs = next_obs

            # ε decay
            if self.eps > self.eps_min:
                self.eps -= self.eps_decay

            # train
            if t > self.batch_size and t % self.train_freq == 0:
                self._train_step()

            # target update
            if t % self.target_update_interval == 0:
                self.q_target.load_state_dict(self.q.state_dict())

            if done:
                obs, _ = self.env.reset()

    def _train_step(self):
        (
            obs,
            act,
            rew,
            next_obs,
            done,
            idx,
            weights,
        ) = self.replay.sample(self.batch_size)

        obs = torch.FloatTensor(obs)
        act = torch.LongTensor(act)
        rew = torch.FloatTensor(rew)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done)
        weights = torch.FloatTensor(weights)

        with torch.no_grad():
            next_q = self.q_target(next_obs).max(1)[0]
            target = rew + self.gamma * (1 - done) * next_q

        q_vals = self.q(obs)[torch.arange(self.batch_size), act]

        td_error = target - q_vals
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_priorities = td_error.abs().detach().numpy() + 1e-6
        self.replay.update_priorities(idx, new_priorities)


# change this the agents !
model = PrioritizedDQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    # learning_starts=0,
    buffer_size=50000,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
)
model.learn(total_timesteps=30000)