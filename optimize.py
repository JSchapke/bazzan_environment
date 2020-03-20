import optuna
from env import Env
from q_learning import QLearning, init_qtables
from ga import build_ga
from ga_ql import ga_ql

'''
'''

def ga_params(trial):
    pass

def ql_params(trial):
    pass

def ga_ql_params(trial):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    # Environment Params
    parser.add_argument('netfile')
    parser.add_argument('--h', help='Max number of hops allowed by the agent.', default=2, type=int)
    parser.add_argument('--k', help='Max number of K shortest routes from an agent to a target.', default=2, type=int)
    # Simulation Params
    parser.add_argument('--episodes', help='Number of episodes for a run a trial.', default=1000,type=int)
    parser.add_argument('--trials', help='Number of trials to optimize.', default=100,type=int)
    parser.add_argument('--alg', help='Algorithm to optimize {ga, ql, ga_ql}.', default='ga_ql', type=str)
    return parser.parse_args()

def main(args):
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
