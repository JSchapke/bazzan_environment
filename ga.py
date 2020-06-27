import os
import argparse
import datetime
import random
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from lib.pyeasyga import GeneticAlgorithm
from env import Env

def create_individual(action_space, *args):
    high = action_space.high
    low = action_space.low
    return [random.randint(low[i], high[i]) for i in range(len(high))]

def fitness(env, actions, *args):
    rewards = env.step(actions)
    return np.mean(rewards)

def mutate(action_space, individual):
    high = action_space.high
    low = action_space.low
    index = random.randrange(len(individual))
    individual[index] = random.randint(low[index], high[index])

def build_ga(env, 
        population, 
        crossover_rate=0,
        mutation_rate=0,
        generations=1,
        elitism=0,
        **kwargs):
    _fitness = partial(fitness, env)
    _mutate = partial(mutate, env.action_space)
    _create_individual = partial(create_individual, env.action_space)

    ga = GeneticAlgorithm(None,
                           population_size=population,
                           generations=generations,
                           crossover_probability=crossover_rate,
                           mutation_probability=mutation_rate,
                           elitism=elitism,
                           maximise_fitness=True )
    ga.create_individual = _create_individual
    ga.mutate_function = _mutate
    ga.fitness_function = _fitness
    return ga


def run_ga(env, n_runs, generations, ga_params):
    rewards = np.zeros((n_runs, generations))
    for r in range(n_runs):
        ga = build_ga(env, **ga_params)
        ga.create_first_generation()

        for g in range(generations):
            ga.create_next_generation()

            reward = ga.best_individual()[0]
            rewards[r, g] = reward

            print(f'Run {r+1}/{n_runs}  -  Generation {g+1}/{generations}  -  Generation Reward: {reward}', end='\r')
    return rewards


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--s', help='Max number of steps allowed from an origin to a destination node.', default=2, type=int)
    parser.add_argument('--k', help='Max number of shortest paths from an origin to a destination node.', default=2, type=int)
    # Agent Params
    parser.add_argument('--generations', '-g', help='Number of generations to run.', default=500, type=int)
    parser.add_argument('--population', '-p', help='Size of population.', default=20, type=int)
    parser.add_argument('--crossover_rate', '-c', help='Rate of crossover.', default=0.5, type=float)
    parser.add_argument('--mutation_rate', '-m', help='Rate of mutation.', default=0.5, type=float)
    parser.add_argument('--elitism', '-e', help='Size of the elite.', default=0)
    # Simulation Params
    parser.add_argument('--runs', help='Number of runs for GA.', default=1, type=int)
    parser.add_argument('--outdir', help='Output dir for the plot.', default='./figs/')
    return parser.parse_args()

if __name__ == '__main__':
    print('- Starting Generic Algorithm -')
    args = parse_args()

    now = datetime.datetime.now()
    hour = ('0' + str(now.hour))[-2:]
    minute = ('0' + str(now.minute))[-2:]
    filename = f'GA_g{args.generations}_p{args.population}_{hour}{minute}.png'
    outpath = os.path.join(args.outdir, filename)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    hists = []
    env = Env(args.netfile, h=args.s, k=args.k)
    action_space = env.action_space


    ga_params = dict(
                population=args.population, 
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                elitism=args.elitism)

    rewards = run_ga(env, args.runs, args.generations, ga_params)

    generations = range(args.generations)
    means = rewards.mean(0)
    stds = rewards.std(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Genetic Algorithm rewards')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Reward')
    ax.plot(generations, means, label='AVG. Generation Reward')
    plt.fill_between(generations, means-stds, means+stds,alpha=0.2)
    legend = ax.legend(loc='lower right', shadow=True)
    plt.savefig(outpath)
    print(f'Figure saved to: {outpath}')

