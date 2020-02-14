import copy
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
            eps,
            decay,
            alpha,
            qtables=None):

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
            self.qtables = copy.deepcopy(qtables)

        self.eps = eps
        self.decay = decay
        self.alpha = alpha


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

    def update(self, actions, rewards):
        self.eps = self.eps * self.decay

        for a, qtable in enumerate(self.qtables):
            action = actions[a]
            reward = rewards[a]
            qtable[action] = qtable[action] + self.alpha * (reward - qtable[action])


def init_qtables(env):
    qtables = []
    for agent in env.all_agents:
        qtable = np.zeros(len(env.choices[agent]))
        for i, route in enumerate(env.choices[agent]):
            qtable[i] = 1 / len(route)
        qtables.append(qtable)


def parse_args():
    parser = argparse.ArgumentParser()
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--h', help='Max number of hops allowed by the agent.', default=2, type=int)
    parser.add_argument('--k', help='Max number of K shortest routes from an agent to a target.', default=2, type=int)
    # Agent Params
    parser.add_argument('--eps', help='Starting epsilon for QLearning.', default=.05, type=float)
    parser.add_argument('--decay', help='Epsilon decay rate.', default=1, type=float)
    parser.add_argument('--alpha', help='Alpha value of QLearning.', default=0.1, type=float)
    # Simulation Params
    parser.add_argument('--episodes', help='Number of episodes for a run of QLearning.', default=2000,type=int)
    parser.add_argument('--runs', help='Number of runs for QLearning.', default=30, type=int)
    parser.add_argument('--outpath', help='Output path for plot.', default='./figs/qlearn.png')
    return parser.parse_args()


if __name__ == '__main__':
    print('- Starting QLearning -')
    args = parse_args()

    env = Env(args.netfile, h=args.h, k=args.k)
    action_space = env.action_space
    print(f'N. Agents: {len(action_space.high)}')

    # initialize qtable with geodesic distances of choices
    qtables = init_qtables(env)

    all_avg_rewards = np.zeros((args.runs, args.episodes))
    episodes = range(args.episodes)

    for r in range(args.runs):
        agent = QLearning(
                    action_space, 
                    qtables=qtables,
                    eps=args.eps,
                    decay=args.decay,
                    alpha=args.alpha )

        for e in episodes:
            actions = agent.act()

            rewards = env.step(actions)

            agent.update(actions, rewards)

            all_avg_rewards[r, e] = rewards.mean() 

            print(f'Run {r+1}/{args.runs}  -  Episode {e+1}/{args.episodes}  -  Episode Reward: {rewards.mean()}', end='\r')

    means = all_avg_rewards.mean(0)
    stds = all_avg_rewards.std(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('QLearning Agent Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot(episodes, means, label='AVG. Agent Reward')
    plt.fill_between(episodes, means-stds, means+stds,alpha=0.2)
    legend = ax.legend(loc='lower right', shadow=True)
    plt.savefig(args.outpath)
    print(f'\nFigure saved to: {args.outpath}')
    #plt.show()