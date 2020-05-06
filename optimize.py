import optuna
from functools import partial

from q_learning import *
from ga import *
from ga_ql import *

def ga_objective(env, trial):
    c = trial.suggest_uniform('cross_rate', 0, 1)
    m = trial.suggest_uniform('mutation_rate', 0, 1)

    ga_params = dict(
                population=100, 
                crossover_rate=c,
                mutation_rate=m,
                elitism=True)

    reward = run_ga(env, 1, 1000, ga_params)
    return -reward.mean()


def ql_objective(env, trial):
    action_space = env.action_space
    qtables = init_qtables(env)

    alpha = trial.suggest_uniform('alpha', 0, 1)
    eps = trial.suggest_uniform('eps', 0, 1)
    decay = trial.suggest_uniform('decay', 0, 1)

    ql_params = dict(
        action_space=action_space, 
        qtables=qtables,
        eps=eps,
        decay=decay,
        alpha=alpha)
    rewards = run_ql(env, 1, 1000, ql_params)
    return -rewards.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('netfile', help='File containing the network')
    parser.add_argument('--algo', help='Which algorithm to be optimized ["ql", "ga"].', default="ql")
    parser.add_argument('--s', help='Max number of steps allowed from an origin to a destination node.', default=2, type=int)
    parser.add_argument('--k', help='Max number of shortest paths from an origin to a destination node.', default=2, type=int)
    args = parser.parse_args()

    env = Env(args.netfile, h=args.s, k=args.k)
    study = optuna.create_study()

    if args.algo == 'ga':
        objective = partial(ga_objective, env)

    elif args.algo == 'ql':
        objective = partial(ql_objective, env)

    study.optimize(objective, n_trials=20)
    print('Best Parameters:')
    print(study.best_params)
