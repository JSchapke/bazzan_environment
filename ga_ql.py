

def parse_args():
    parser = argparse.ArgumentParser()
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--h', help='Max number of hops allowed by the agent.', default=2, type=int)
    parser.add_argument('--k', help='Max number of K shortest routes from an agent to a target.', default=2, type=int)
    # GA Params
    parser.add_argument('--generations', help='Number of generations to run.', default=50, type=int)
    parser.add_argument('--population', help='Size of population.', default=100, type=int)
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
    print(f'Starting GA. Running for {args.generations} generations with {args.population} individuals.')
    ga.run()

    reward, actions = ga.best_individual()
    routes = env.get_routes(actions)

    print('---- Results ----')
    print('Best reward:', reward)
    print('Routes:')
    for route in routes:
        print(route)
    print('-----------------')

#    agent_avg_rewards = []
#    system_rewards    = []
#    episodes = range(args.episodes)
#    for e in episodes:
#        actions = agent.act()
#
#        rewards = env.step(actions)
#
#        agent.save(actions, rewards)
#
#        agent_avg_rewards.append(rewards.mean())
#        system_rewards.append(rewards.sum())
#
#    fig, ax = plt.subplots(figsize=(8, 6))
#    ax.set_title('QLearning Agent Rewards')
#    ax.plot(episodes, agent_avg_rewards, label='AVG. Agent Reward')
#    legend = ax.legend(loc='lower right', shadow=True)
#    plt.show()
#
#    fig, ax = plt.subplots(figsize=(8, 6))
#    ax.set_title('QLearning System Reward')
#    ax.plot(episodes, system_rewards, label='Sum of agents reward.')
#    legend = ax.legend(loc='lower right', shadow=True)
#    plt.show()
