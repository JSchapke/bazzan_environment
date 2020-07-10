import os
import gym
import igraph
import numpy as np
from itertools import chain
import time
from collections import defaultdict

from lib.utils import read_graph, get_agent_routes

class Env:
    def __init__(self, graphfile, k=1, h=2):
        '''
        graphfile - String, path to graphfile
        k - Integer, k shortest paths to be considered
        h - Integer, h number of hops to be considered
        '''
        self.games_played = 0
        self.graphfile = graphfile
        self.h = h
        self.k = k

        self.read_graph()
        self.setup() 

    def read_graph(self):
        if not os.path.isfile(self.graphfile): 
            raise Exception('File not found!')

        G, agents, n_agents, targets, functions = read_graph(self.graphfile)
        self.G = G
        self.agents = agents
        self.n_agents = n_agents
        self.targets = targets
        self.functions = functions

    def write_graph(self):
        pass

    def show_graph(self):
        visual_style = {}
        self.G.vs["label"] = self.G.vs["name"]

        # Define colors used for outdegree visualization
        colours = ['#fecc5c', '#a31a1c']

        # Set bbox and margin
        visual_style["bbox"] = (1000,1000)
        visual_style["margin"] = 50
        visual_style["vertex_color"] = 'grey'
        visual_style["vertex_size"] = 50
        visual_style["vertex_label_size"] = 20
        visual_style["edge_curved"] = False

        # Set the layout
        my_layout = self.G.layout_fruchterman_reingold()
        visual_style["layout"] = my_layout


        # Plot the graph
        igraph.plot(self.G, **visual_style)

    
    def setup(self):
        self.choices = []
        self.n_choices = []
        self.agent_targets = []

        for a, agent in enumerate(self.agents):
            routes = get_agent_routes(self.G, agent, h=self.h, k=self.k)
            targets = list(routes.keys())
            v = routes.values()
            choices = list(chain.from_iterable(v))
            n_choices = len(choices)

            self.agent_targets.append(targets)
            self.choices.append(choices)
            self.n_choices.append(n_choices)

        agents = []
        for i, n in enumerate(self.n_agents):
            agents += [i] * n
        self.all_agents = np.array(agents).flatten()

        low = np.zeros(len(self.all_agents))
        high = np.array([self.n_choices[a] for a in self.all_agents]) - 1
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=int)
        self.observation_space = None
        
        names = self.G.vs['name']
        ids = range(len(names))
        self.node_mapping = dict(zip(names, ids))

    def get_routes(self, actions):
        '''
        Returns a list with the routes taken by each agent in the given action list
        '''
        routes = []
        for i, action in enumerate(actions):
            agent = self.all_agents[i]
            routes.append(self.choices[agent][action])
        return routes

    def test(self, i):
        return i 


    def step(self, actions):
        if len(actions) != len(self.all_agents):
            raise Exception('Number of actions does not match number of agents.')
        self.games_played += 1

        paths = []
        edges = {}
        
        targets, target_flows = [], {}
        values  = np.zeros(len(self.all_agents))
        costs   = np.zeros(len(self.all_agents))

        cur = 0
        agent_action = {}
        all_acts = []
        for a, n in enumerate(self.n_agents):
            acts, counts = np.unique(actions[cur:cur+n], return_counts=True)
            all_acts.append(acts)
            cur += n

            for action, count in zip(acts, counts):
                if not self.legal_action(a, action):
                    raise ValueError(f'Illegal action {action} for agent {a}.')
         
                path = self.choices[a][action]
                path_ids = [self.node_mapping[node] for node in path]
                path_edges = self.G.get_eids(path=path_ids)
                edges_, counts_ = np.unique(path_edges, return_counts=True)
                for e, c in zip(edges_, counts_):
                    edges[e] = c * count if e not in edges else edges[e] + c * count

                p = path[-1]
                targets.append(p)
                target_flows[p] = target_flows[p] + count if p in target_flows else count

                agent_action[(a, action)] = {
                        'path': path_edges,
                        'edges': edges_,
                        'target': p }

        flows = list(edges.values())
        edges = list(edges.keys())
        edge_flow = dict(zip(edges, flows))

        _values = {}
        _costs = {}

        for k, v in agent_action.items():
            cost = 0
            path = v['path']
            for edge in path:
                flow = edge_flow[edge]
                cf = self.G.es[edge]['cost_fn']
                cost += self.functions[cf](flow)

            t = v['target']
            flow = target_flows[t]
            vf = self.G.vs[self.node_mapping[t]]['value']
            _values[k] = self.functions[vf](flow)
            _costs[k]  = cost

        
        cur = 0
        for a, n in enumerate(self.n_agents):
            acts = actions[cur:cur+n]
            for act in all_acts[a]:
                mask = actions[cur:cur+n] == act
                values[cur:cur+n][mask] = _values[(a,act)]
                costs[cur:cur+n][mask] = _costs[(a,act)]
            cur += n 


        rewards = values - costs
        return rewards


    def legal_action(self, agent, action):
        low = 0
        high = self.n_choices[agent]
        if action < low or action > high:
            print(f'Invalid action {action} for agent {agent} bounds: ({low}, {high})')
            return False
        return True

if __name__ == '__main__':

    env = Env('./envs/braessb3_4200.txt', k=8, h=10)
    print('Environment built.\n')
    print('env.action_space:', env.action_space)
    print('env.agents:', env.agents)
    print('env.targets:', env.targets)

    low = env.action_space.low
    high = env.action_space.high
    num_actions = [high[i]-low[i]+1 for i in range(len(env.agents))]
    print('N. possible actions per agent:', [(agent, num_actions[a]) for a, agent in enumerate(env.agents)])
    
    env.show_graph()

