import os
import argparse
import random
from functools import partial
from pyeasyga import pyeasyga
import numpy as np
import matplotlib.pyplot as plt

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
        eletism=False,
        **kwargs):
    _fitness = partial(fitness, env)
    _mutate = partial(mutate, env.action_space)
    _create_individual = partial(create_individual, env.action_space)

    ga = pyeasyga.GeneticAlgorithm(None,
                                   population_size=population,
                                   generations=generations,
                                   crossover_probability=crossover_rate,
                                   mutation_probability=mutation_rate,
                                   elitism=eletism,
                                   maximise_fitness=True )
    ga.create_individual = _create_individual
    ga.mutate_function = _mutate
    ga.fitness_function = _fitness
    return ga


def parse_args():
    parser = argparse.ArgumentParser()
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--h', help='Max number of hops allowed by the agent.', default=2, type=int)
    parser.add_argument('--k', help='Max number of K shortest routes from an agent to a target.', default=2, type=int)
    # Agent Params
    parser.add_argument('--generations', help='Number of generations to run.', default=2000, type=int)
    parser.add_argument('--population', help='Size of population.', default=5, type=int)
    parser.add_argument('--crossover_rate', help='Rate of crossover.', default=0.5, type=float)
    parser.add_argument('--mutation_rate', help='Rate of mutation.', default=0.5, type=float)
    parser.add_argument('--eletism', help='Keep the most fit individual in the population.', default=False, action='store_true')
    # Simulation Params
    parser.add_argument('--runs', help='Number of runs for GA.', default=30, type=int)
    parser.add_argument('--outpath', help='Output path for plot.', default='./figs/ga.png')
    return parser.parse_args()

if __name__ == '__main__':
    print('- Starting Generic Algorithm -')
    args = parse_args()

    hists = []
    env = Env(args.netfile, h=args.h, k=args.k)
    action_space = env.action_space

    rewards = np.zeros((args.runs, args.generations))
    generations = range(args.generations)

    for r in range(args.runs):
        ga = build_ga(env,
                args.population, 
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                eletism=args.eletism)
        ga.create_first_generation()

        for g in generations:
            ga.create_next_generation()

            cur_gen = ga.current_generation
            reward = np.mean([i.fitness for i in cur_gen])
            rewards[r, g] = reward

            print(f'Run {r+1}/{args.runs}  -  Generation {g+1}/{args.generations}  -  Generation Reward: {reward}', end='\r')

    means = rewards.mean(0)
    stds = rewards.std(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Genetic Algorithm rewards')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Reward')
    ax.plot(generations, means, label='AVG. Generation Reward')
    plt.fill_between(generations, means-stds, means+stds,alpha=0.2)
    legend = ax.legend(loc='lower right', shadow=True)
    plt.savefig(args.outpath)
    print(f'Figure saved to: {args.outpath}')

#    routes = env.get_routes(actions)
#    print('---- Results ----')
#    print('Best reward:', reward)
#    print('Routes:')
#    for route in routes:
#        print(route)
#    print('-----------------')

