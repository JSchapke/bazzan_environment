import gym
import igraph
import numpy as np

from utils import read_graph

class Env:
    def __init__(self, graphfile, k=1, h=2):
        '''
        graphfile - String, path to graphfile
        k - Integer, k shortest paths to be considered
        h - Integer, h number of hops to be considered
        '''
        self.graphfile = graphfile
        self.h = h
        self.k = k

        self.read_graph()
        #read_graph(graphfile)
        self.setup() 

    def read_graph(self):
        ''' sample graph '''
        self.cost_functions = [lambda f: 5, lambda f: f, lambda f: 2*f]
        G = igraph.Graph()
        G.add_vertices(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'])

        self.agents = np.array(['A1', 'A2', 'A3', 'A4', 'A5', 'A6'])
        G.vs['agent'] = np.array([True] * 6 + [False] * 6)

        self.targets =   np.array(['M1', 'M2', 'M3', 'M4', 'M5', 'M6'])
        G.vs['agent'] = np.array([False] * 6 + [True] * 6)
        G.vs['value'] =  np.array([None] * 6 + [10, 13, 10, 10, 13, 10])

        G.add_edge('A1', 'M1', cost_fn=0)
        G.add_edge('A1', 'A2', cost_fn=1)
        G.add_edge('A2', 'M2', cost_fn=2)
        G.add_edge('A2', 'A3', cost_fn=1)
        G.add_edge('A3', 'M3', cost_fn=0)
        G.add_edge('A3', 'A4', cost_fn=1)
        G.add_edge('A4', 'M4', cost_fn=0)
        G.add_edge('A4', 'A5', cost_fn=1)
        G.add_edge('A5', 'M5', cost_fn=2)
        G.add_edge('A5', 'A6', cost_fn=1)
        G.add_edge('A6', 'M6', cost_fn=0)
        self.G = G
    
    def setup(self):
        self.choices = []
        self.n_choices = []
        self.agent_targets = []

        distances = self.G.shortest_paths(self.agents, self.targets)
        self.distances = np.array(distances)
        
        for a, agent in enumerate(self.agents):
            _distances = self.distances[a]
            targets = self.targets[_distances <= self.h]

            #FIXME find k shortest routes
            choices = self.G.get_shortest_paths(agent, targets)
            n_choices = len(choices) 

            self.agent_targets.append(targets)
            self.choices.append(choices)
            self.n_choices.append(n_choices)

        low = np.zeros(len(self.agents))
        high = np.array(self.n_choices) - 1
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=int)
        self.observation_space = None


    def find_k_paths(self, agent, target):
        pass

    def step(self, actions):
        paths = []
        edges = np.array([], dtype=int)
        
        values  = np.zeros(len(self.agents))
        costs   = np.zeros(len(self.agents))
        
        for i, action in enumerate(actions):
            if not self.legal_action(i, action):
                rewards[i] = None
                paths.append(None)
                continue

            path = self.choices[i][action]
            path_edges = self.G.get_eids(path=path)
            paths.append(path_edges)
            edges_ = np.unique(path_edges)
            edges = np.append(edges, edges_)

            value = self.G.vs[path[-1]]['value']
            values[i] = value

        edges, flows = np.unique(edges, return_counts=True)
        flows = dict(zip(edges, flows))

        for i, path in enumerate(paths):
            if path is None:
                continue
            cost = 0
            for edge in path:
                flow = flows[edge]
                cf = self.G.es[edge]['cost_fn']
                cost += self.cost_functions[cf](flow)

            costs[i] = cost

        rewards = values - costs
        return rewards

    def legal_action(self, agent, action):
        low = self.action_space.low[agent]
        high = self.action_space.high[agent]
        if action < low or action > high:
            print(f'Invalid action {action}. Agent {agent} bounds: ({low}, {high})')
            return False
        return True

if __name__ == '__main__':
    env = Env('./env0.txt')
    print('Environment built.\n')
    print('env.action_space:', env.action_space)
    print('env.agents:', env.agents)
    print('env.targets:', env.targets)

    low = env.action_space.low
    high = env.action_space.high
    num_actions = [high[i]-low[i]+1 for i in range(len(env.agents))]
    print('N. possible actions per agent:', [(agent, num_actions[a]) for a, agent in enumerate(env.agents)])

    print('-'*10, 'Test Cases', '-'*10)
    for i in range(5):
        print(f'Test Run N.{i}:')
        actions = []
        for a, agent in enumerate(env.agents):
            action = np.random.randint(low[a], high[a]+1)
            actions.append(action)
        print('Chosen actions:', actions)

        reward = env.step(actions)
        total_reward = sum(reward)

        print('Rewards:', reward)
        print('Total Reward:', total_reward, '\n')
    print('-'*35)

    system_optimum_actions = [0, 1, 1, 1, 1, 1]
    system_optimum_rewards = env.step(system_optimum_actions)
    system_optimum = sum(system_optimum_rewards)
    print('\nSystem Optimum:')
    print('Actions under system optimum:', system_optimum_actions)
    print('Individual rewards under system optimum:', system_optimum_rewards)
    print('Total reward:', system_optimum)

    ue_actions = [1, 1, 0, 2, 1, 0]
    ue_rewards = env.step(ue_actions)
    ue = sum(ue_rewards)
    print('\nUser Equilibrium:')
    print('Actions under user equilibrium:', ue_actions)
    print('Individual rewards under user equilibrium:', ue_rewards)
    print('Total reward:', ue)

