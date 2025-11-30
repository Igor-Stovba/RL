
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import random
from collections import deque

class QLAgent:

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0
        

    def act(self):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward



class QLAgentDouble:

    def __init__(self, starting_state, state_space, action_space,
                 alpha=0.5, gamma=0.95,
                 exploration_strategy=EpsilonGreedy()):

        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.acc_reward = 0

        self.q1 = {starting_state: [0 for _ in range(action_space.n)]}
        self.q2 = {starting_state: [0 for _ in range(action_space.n)]}
        

    @property
    def q_table(self):
        return {
            s: [(self.q1[s][i] + self.q2[s][i]) / 2 for i in range(self.action_space.n)]
            for s in set(self.q1.keys()) | set(self.q2.keys())
        }

    def act(self):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        for q in (self.q1, self.q2):
            if next_state not in q:
                q[next_state] = [0 for _ in range(self.action_space.n)]

        s, s1, a = self.state, next_state, self.action

        if random.random() < 0.5:
            best_a = max(range(self.action_space.n), key=lambda x: self.q1[s1][x])
            target = reward + self.gamma * self.q2[s1][best_a]
            self.q1[s][a] += self.alpha * (target - self.q1[s][a])
        else:
            best_a = max(range(self.action_space.n), key=lambda x: self.q2[s1][x])
            target = reward + self.gamma * self.q1[s1][best_a]
            self.q2[s][a] += self.alpha * (target - self.q2[s][a])

        self.state = s1
        self.acc_reward += reward
        

        
class QLAgentReplay:

    def __init__(self, starting_state, state_space, action_space,
                 alpha=0.5, gamma=0.95, buffer_size=1000, batch_size=32,
                 exploration_strategy=EpsilonGreedy()):

        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.acc_reward = 0

        self.q_table = {starting_state: [0 for _ in range(action_space.n)]}

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def act(self):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def _store(self, s, a, r, s1, done):
        self.buffer.append((s, a, r, s1, done))

    def _sample_batch(self):
        return random.sample(self.buffer, min(len(self.buffer), self.batch_size))

    def learn(self, next_state, reward, done=False):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        self._store(self.state, self.action, reward, next_state, done)

        if len(self.buffer) >= self.batch_size:
            batch = self._sample_batch()
            for s, a, r, s1, d in batch:
                if s1 not in self.q_table:
                    self.q_table[s1] = [0 for _ in range(self.action_space.n)]

                target = r + (0 if d else self.gamma * max(self.q_table[s1]))
                self.q_table[s][a] += self.alpha * (target - self.q_table[s][a])

        self.state = next_state
        self.acc_reward += reward