import optuna

from q_learning import *
from ga import *
from ga_ql import *

def run_ql(env, episodes, eps, decay, alpha):

    action_space = env.action_space

    qtables = init_qtables(env)

    agent = QLearning(
                action_space, 
                qtables=qtables,
                eps=eps,
                decay=decay,
                alpha=alpha )

    for e in episodes:
        actions = agent.act()
        rewards = env.step(actions)
        agent.update(actions, rewards)

    return rewards.mean()

def run_ga(env, episodes, eps, decay, alpha):

    action_space = env.action_space

    qtables = init_qtables(env)

    agent = QLearning(
                action_space, 
                qtables=qtables,
                eps=eps,
                decay=decay,
                alpha=alpha )

    for e in episodes:
        actions = agent.act()
        rewards = env.step(actions)
        agent.update(actions, rewards)

    return rewards.mean()


def run_gaql(env, episodes, eps, decay, alpha):
    delta
    population
    crossover_rate
    mutation_rate
    eletism
    epsilon
    decay
    alpha

    action_space = env.action_space

    qtables = init_qtables(env)

    agent = QLearning(
                action_space, 
                qtables=qtables,
                eps=eps,
                decay=decay,
                alpha=alpha )

    for e in episodes:
        actions = agent.act()
        rewards = env.step(actions)
        agent.update(actions, rewards)

    return rewards.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('netfile', help='File containing the network')
    parser.add_argument('--algo', help='Which algorithm to be optimized ["ql", "ga", "gaql"].', default="ql")
    parser.add_argument('--s', help='Max number of steps allowed from an origin to a destination node.', default=2, type=int)
    parser.add_argument('--k', help='Max number of shortest paths from an origin to a destination node.', default=2, type=int)
    parser.parse_args()

    def

