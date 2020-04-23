import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from env import Env
from q_learning import QLearning, init_qtables
from ga import build_ga

def ga_ql(args, env, run=None):

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

    ga_history = np.zeros(args.episodes)
    ql_history = np.zeros(args.episodes)

    for tau in range(args.episodes):
        if tau > 0 and tau % args.delta == 0:
            actions = best_solution 
        else:
            actions = agents.act()

        # QL step
        rewards = env.step(actions)
        agents.update(actions, rewards)

        current_solution = actions 

        # Adds current solution to GA
        authority.current_generation[-1].genes = current_solution
        authority.current_generation[-1].fitness = rewards.mean()

        # GA step
        authority.create_next_generation()
        best_solution = authority.best_individual()[1]

        ql_history[tau] = rewards.mean()
        ga_gen = authority.current_generation
        ga_history[tau] = np.mean([i.fitness for i in ga_gen])

        print(f'Run {r+1}/{args.runs}  -  Episode {tau+1}/{args.episodes}  -  Episode Reward: {rewards.mean()}', end='\r')
    return best_solution, ql_history, ga_history


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--s', help='Max number of steps taken by an agent.', default=2, type=int)
    parser.add_argument('--k', help='Max number of shortest paths from an origin to a destination node.', default=2, type=int)
    # GA Params
    parser.add_argument('--delta', help='Frequency of interaction GA -> QL.', default=10, type=int)
    parser.add_argument('--population', '-p', help='Size of population.', default=100, type=int)
    parser.add_argument('--crossover_rate', '-c', help='Rate of crossover.', default=0.5, type=float)
    parser.add_argument('--mutation_rate', '-m', help='Rate of mutation.', default=0.5, type=float)
    parser.add_argument('--eletism', '-e', help='Keep the most fit individual in the population. Parameter does not take any arguments.', default=False, action='store_true')
    # QLearn Params
    parser.add_argument('--epsilon', '-eps', help='Starting epsilon for QLearning.', default=.05, type=float)
    parser.add_argument('--decay', help='Epsilon decay rate.', default=1, type=float)
    parser.add_argument('--alpha', help='Learning rate alpha value of QLearning.', default=0.5, type=float)
    # Simulation Params
    parser.add_argument('--runs', help='Number of runs for GA-QL.', default=1, type=int)
    parser.add_argument('--episodes', help='Number of episodes to run.', default=500, type=int)
    parser.add_argument('--outdir', help='Output dir for the plot.', default='./figs/')
    return parser.parse_args()

if __name__ == '__main__':
    print('- Starting GA-QL -')
    args = parse_args()

    now = datetime.datetime.now()
    hour = ('0' + str(now.hour))[-2:]
    minute = ('0' + str(now.minute))[-2:]
    filename = f'GaQL_a{args.alpha}_p{args.population}_d{args.delta}_{hour}{minute}.png'
    outpath = os.path.join(args.outdir, filename)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    env = Env(args.netfile, h=args.s, k=args.k)

    ql_rewards = np.zeros((args.runs, args.episodes))
    ga_rewards = np.zeros((args.runs, args.episodes))
    for r in range(args.runs):
        best_solution, ql_history, ga_history = ga_ql(args, env, run=r)
        ql_rewards[r] = ql_history
        ga_rewards[r] = ga_history
    print('')

    ga_means = ga_rewards.mean(0)
    ga_stds = ga_rewards.std(0)
    ql_means = ql_rewards.mean(0)
    ql_stds = ql_rewards.std(0)
    episodes = range(args.episodes)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('GA-QL Agent Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot(episodes, ga_means, label='GA under GA-QL.')
    plt.fill_between(episodes, ga_means-ga_stds, ga_means+ga_stds,alpha=0.2)
    ax.plot(episodes, ql_means, label='QL under GA-QL.')
    plt.fill_between(episodes, ql_means-ql_stds, ql_means+ql_stds,alpha=0.2)
    legend = ax.legend(loc='lower right', shadow=True)
    plt.savefig(outpath)
    print(f'Figure saved to: {outpath}')
