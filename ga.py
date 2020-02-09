import os
import argparse
import random
from functools import partial
from pyeasyga import pyeasyga
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from env import Env

def create_individual(action_space, *args):
    high = action_space.high
    low = action_space.low
    return [random.randint(low[i], high[i]) for i in range(len(high))]

def fitness(env, history, actions, *args, iterable=None):
    rewards = env.step(actions)
    history.append(rewards)

    if iterable is not None:
        iterable.update()
        #tqdm.set_description(iterable['iter'], desc='Loss: %.4f. Train Accuracy %.4f. Validation Accuracy: %.4f' % (loss, train_acc, val_acc), refresh=True)

    return np.sum(rewards)

def mutate(action_space, individual):
    high = action_space.high
    low = action_space.low
    index = random.randrange(len(individual))
    individual[index] = random.randint(low[index], high[index])

def parse_args():
    parser = argparse.ArgumentParser()
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--h', help='Max number of hops allowed by the agent.', default=2, type=int)
    parser.add_argument('--k', help='Max number of K shortest routes from an agent to a target.', default=2, type=int)
    # Agent Params
    parser.add_argument('--generations', help='Number of generations to run.', default=10, type=int)
    parser.add_argument('--population', help='Size of population.', default=50, type=int)
    parser.add_argument('--crossover_rate', help='Rate of crossover.', default=0.5, type=float)
    parser.add_argument('--mutation_rate', help='Rate of mutation.', default=0.5, type=float)
    parser.add_argument('--eletism', help='Keep the most fit individual in the population.', default=False, action='store_true')

    parser.add_argument('--outpath', help='Output directory for plot.', default='./')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(args.outpath):
        raise Exception(f'Directory "{args.outpath}" does not exist.')

    env = Env(args.netfile, h=args.h, k=args.k)
    action_space = env.action_space

    history = []
    fitness = partial(fitness, env, history)
    mutate = partial(mutate, action_space)
    create_individual = partial(create_individual, action_space)

    ga = pyeasyga.GeneticAlgorithm(None,
                                   population_size=args.population,
                                   generations=args.generations,
                                   crossover_probability=args.crossover_rate,
                                   mutation_probability=args.mutation_rate,
                                   elitism=args.eletism,
                                   maximise_fitness=True )
    ga.create_individual = create_individual
    ga.mutate_function = mutate
    ga.fitness_function = fitness
    print(f'Starting GA. Running for {args.generations} generations with {args.population} populations.')
    ga.run()

    reward, actions = ga.best_individual()
    routes = env.get_routes(actions)

    print('---- Results ----')
    print('Best reward:', reward)
    print('Routes:')
    for route in routes:
        print(route)
    print('-----------------')

    history = np.stack(history)
    agent_avg_rewards = history.mean(1)
    system_rewards = history.sum(1)
    generations = range(args.generations * args.population)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Genetic Algorithm System Reward')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Reward')
    ax.plot(generations, system_rewards)
    plt.show()
