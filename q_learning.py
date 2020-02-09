import argparse
import os
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

from env import Env

class QLearning:
    def __init__(self, 
            action_space, 
            qtables=None,
            eps=1,
            decay=0.9,
            min_eps=0.01,
            lr=0.1,
            maxlen=1000,
            batch_size=8,
            update_every=10 ):

        self.n_agents = action_space.shape[0]
        self.high = action_space.high
        self.low = action_space.low

        if qtables is None:
            self.qtables = []
            for a in range(len(self.high)):
                n = self.high[a] - self.low[a] + 1
                qtable = np.zeros(n)
                self.qtables.append(qtable)
        else:
            self.qtables = qtables

        self.eps = eps
        self.decay = decay
        self.min_eps = min_eps
        self.lr = lr
        self.memory = deque(maxlen=1000)
        self.batch_size = batch_size
        self.update_every = update_every
        self.t = 0

    def act(self):
        actions = []
        for agent in range(self.n_agents):
            if np.random.rand() < self.eps:
                l = self.low[agent]
                h = self.high[agent] + 1
                actions.append(np.random.randint(l, h))
            else:
                maxQ = np.max(self.qtables[agent])
                idxs = np.where(self.qtables[agent] == maxQ)[0]
                action = np.random.choice(idxs)
                actions.append(action)
        return actions

    def save(self, actions, rewards):
        self.memory.append((actions, rewards))

        self.t = (self.t + 1) % self.update_every
        if self.t == 0:
            self.update()


    def update(self):
        self.eps = max(self.min_eps, self.eps * self.decay)

        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            for actions, rewards in batch:
                for a, qtable in enumerate(self.qtables):
                    action = actions[a]
                    reward = rewards[a]
                    qtable[action] = qtable[action] + self.lr * (reward - qtable[action])

def parse_args():
    parser = argparse.ArgumentParser()
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--h', help='Max number of hops allowed by the agent.', default=2, type=int)
    parser.add_argument('--k', help='Max number of K shortest routes from an agent to a target.', default=2, type=int)
    parser.add_argument('--episodes', help='Number of episodes to run.', default=500, type=int)
    # Agent Params
    parser.add_argument('--eps', help='Starting epsilon for QLearning.', default=1.0, type=float)
    parser.add_argument('--min_eps', help='Minimum epsilon for QLearning.', default=0.01, type=float)
    parser.add_argument('--decay', help='Epsilon decay rate.', default=0.9, type=float)
    parser.add_argument('--lr', help='Learning rate of QLearning.', default=0.1, type=float)
    parser.add_argument('--maxlen', help='Max number of episodes to storage for QLearning.', default=1000, type=int)
    parser.add_argument('--batch_size', help='Batch size of QLearning.', default=8, type=int)
    parser.add_argument('--update_every', help='Update reate of QLearning.', default=10, type=int)

    parser.add_argument('--outpath', help='Output directory for plot.', default='./')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(args.outpath):
        raise Exception(f'Directory "{args.outpath}" does not exist.')

    env = Env(args.netfile, h=args.h, k=args.k)
    action_space = env.action_space

    # initialize qtable with geodesic distances of choices
    qtables = []
    for agent in range(action_space.shape[0]):
        qtable = np.zeros(len(env.choices[agent]))
        for i, route in enumerate(env.choices[agent]):
            qtable[i] = 1 / len(route)
        qtables.append(qtable)

    agent = QLearning(
                action_space, 
                qtables=qtables,
                eps=args.eps,
                decay=args.decay,
                min_eps=args.min_eps,
                lr=args.lr,
                maxlen=args.maxlen,
                batch_size=args.batch_size,
                update_every=args.update_every )

    agent_avg_rewards = []
    system_rewards    = []
    episodes = range(args.episodes)
    for e in episodes:
        actions = agent.act()

        rewards = env.step(actions)

        agent.save(actions, rewards)

        agent_avg_rewards.append(rewards.mean())
        system_rewards.append(rewards.sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('QLearning Agent Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot(episodes, agent_avg_rewards, label='AVG. Agent Reward')
    legend = ax.legend(loc='lower right', shadow=True)
    plt.savefig('fig.png')
    plt.show()


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('QLearning System Reward')
    ax.plot(episodes, system_rewards, label='Sum of agents reward.')
    legend = ax.legend(loc='lower right', shadow=True)
    #plt.show()





