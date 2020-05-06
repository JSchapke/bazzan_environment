import copy
import argparse
import os
import datetime
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

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
        self.max = self.high.max()

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
        actions = np.zeros(self.n_agents, dtype=int)

        random_actions = np.random.rand(self.n_agents) * (self.high + 1)
        random_actions = np.floor(random_actions)

        probs = np.random.rand(self.n_agents)
        rand_mask = probs < self.eps
        actions[rand_mask] = random_actions[rand_mask]
        
        play_idx = np.where(~rand_mask)[0]
        actions[play_idx] = [self.qtables[i].argmax() for i in play_idx]
        return list(actions)

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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment Params
    parser.add_argument('netfile', help='File containing the network')
    parser.add_argument('--s', help='Max number of steps allowed from an origin to a destination node.', default=2, type=int)
    parser.add_argument('--k', help='Max number of shortest paths from an origin to a destination node.', default=2, type=int)
    # Agent Params
    parser.add_argument('--eps', help='Starting epsilon for QLearning.', default=.05, type=float)
    parser.add_argument('--decay', help='Epsilon decay rate.', default=1, type=float)
    parser.add_argument('--alpha', help='Learning rate alpha value of QLearning.', default=0.5, type=float)
    # Simulation Params
    parser.add_argument('--episodes', help='Number of episodes for a run of QLearning.', default=1000,type=int)
    parser.add_argument('--runs', help='Number of runs for QLearning.', default=1, type=int)
    parser.add_argument('--outdir', help='Output dir for the plot.', default='./figs/')
    return parser.parse_args()


def run_ql(env, n_runs, episodes, ql_params):
    all_avg_rewards = np.zeros((n_runs, episodes))

    for r in range(n_runs):
        agent = QLearning(**ql_params)

        for e in range(episodes):
            actions = agent.act()

            rewards = env.step(actions)

            agent.update(actions, rewards)

            all_avg_rewards[r, e] = rewards.mean() 

            print(f'Run {r+1}/{n_runs}  -  Episode {e+1}/{episodes}  -  Episode Reward: {rewards.mean()}', end='\r')

    return all_avg_rewards


if __name__ == '__main__':
    print('- Starting QLearning -')
    args = parse_args()

    now = datetime.datetime.now()
    hour = ('0' + str(now.hour))[-2:]
    minute = ('0' + str(now.minute))[-2:]
    filename = f'QL_a{args.alpha}_k{args.k}_{hour}{minute}.png'
    outpath = os.path.join(args.outdir, filename)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    env = Env(args.netfile, h=args.s, k=args.k)
    action_space = env.action_space
    print(f'N. Agents: {len(action_space.high)}')

    # initialize qtable with geodesic distances of choices
    episodes = range(args.episodes)
    
    qtables = init_qtables(env)
    params = dict(
        action_space=action_space, 
        qtables=qtables,
        eps=args.eps,
        decay=args.decay,
        alpha=args.alpha)
    all_avg_rewards = run_ql(env, args.n_runs, episodes, params)


    means = all_avg_rewards.mean(0)
    stds = all_avg_rewards.std(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('QLearning Agent Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot(episodes, means, label='AVG. Agent Reward')
    plt.fill_between(episodes, means-stds, means+stds,alpha=0.2)
    legend = ax.legend(loc='lower right', shadow=True)
    plt.savefig(outpath)
    print(f'\nFigure saved to: {outpath}')
