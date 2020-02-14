import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from env import Env
from q_learning import QLearning, init_qtables
from ga import build_ga

def GA_QL(args, env, run=None):

    qtables = init_qtables(env)
    agents = QLearning(
                env.action_space,
                args.epsilon,
                args.decay,
                args.alpha,
                qtables=qtables)

    authority = build_ga(env,
                    args.population, 
                    crossover_rate=args.crossover_rate,
                    mutation_rate=args.mutation_rate,
                    eletism=args.eletism )
    authority.create_first_generation()

    history = np.zeros(args.episodes)

    for tau in range(args.episodes):
        if tau > 0 and tau % args.delta == 0:
            actions = best_solution 
        else:
            actions = agents.act()

        rewards = env.step(actions)
        agents.update(actions, rewards)

        current_solution = actions 
        history[tau] = rewards.mean()

        # Adds current solution to GA
        authority.current_generation[-1].genes = current_solution
        authority.current_generation[-1].fitness = rewards.mean()

        # GA step
        authority.create_next_generation()
        best_solution = authority.best_individual()[1]

        print(f'Run {r+1}/{args.runs}  -  Episode {tau+1}/{args.episodes}  -  Episode Reward: {rewards.mean()}', end='\r')
    return best_solution, history


def parse_args():
    parser = argparse.ArgumentParser()
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--h', help='Max number of hops allowed by the agent.', default=2, type=int)
    parser.add_argument('--k', help='Max number of K shortest routes from an agent to a target.', default=2, type=int)
    # GA Params
    parser.add_argument('--delta', help='Frequency of interaction GA -> QL.', default=4, type=int)
    parser.add_argument('--population', help='Size of population.', default=10, type=int)
    parser.add_argument('--crossover_rate', help='Rate of crossover.', default=0.5, type=float)
    parser.add_argument('--mutation_rate', help='Rate of mutation.', default=0.5, type=float)
    parser.add_argument('--eletism', help='Keep the most fit individual in the population.', default=False, action='store_true')
    # QLearn Params
    parser.add_argument('--epsilon', help='Starting epsilon for QLearning.', default=.05, type=float)
    parser.add_argument('--decay', help='Epsilon decay rate.', default=1, type=float)
    parser.add_argument('--alpha', help='Alpha value of QLearning.', default=0.1, type=float)
    # Simulation Params
    parser.add_argument('--runs', help='Number of runs for GA-QL.', default=30, type=int)
    parser.add_argument('--episodes', help='Number of episodes to run.', default=2000, type=int)
    parser.add_argument('--outpath', help='Output directory for plot.', default='./figs/ga_ql.png')
    return parser.parse_args()

if __name__ == '__main__':
    print('- Starting GA-QL -')
    args = parse_args()

    env = Env(args.netfile, h=args.h, k=args.k)

    rewards = np.zeros((args.runs, args.episodes))
    for r in range(args.runs):
        best_solution, history = GA_QL(args, env, run=r)
        rewards[r] = history
    print('')

    means = rewards.mean(0)
    stds = rewards.std(0)
    episodes = range(args.episodes)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('GA_QL Agent Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot(episodes, means, label='AVG. Agent Reward')
    plt.fill_between(episodes, means-stds, means+stds,alpha=0.2)
    legend = ax.legend(loc='lower right', shadow=True)
    plt.savefig(args.outpath)
    print(f'Figure saved to: {args.outpath}')
