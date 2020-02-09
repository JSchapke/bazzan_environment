from collections import defaultdict
from heapq import heappush, heappop
import re

import numpy as np
import igraph

re_cost_function = r'^#\scost_functions'
re_edge = r'^#\sedges'
re_target = r'^#\stargets'
re_agent = r'^#\sagents'

def read_graph(path):
    edges, edge_fn = [], []
    targets, target_fn = [], []
    agents = []
    n_agents = []
    functions = {}

    reading = ''
    f = open(path, 'r')
    for line in f.readlines():
        if line.strip() == '':
            reading = ''
            continue
        if reading == 'cost_functions':
            name, fun = line.split('=')
            functions[name.strip()] = eval(fun.strip())
        elif reading == 'edges':
            A, B, FN = line.strip().split(' ')
            edges.append((A, B))
            edge_fn.append(FN)
        elif reading == 'agents':
            A, num = line.strip().split(' ')
            agents.append(A)
            n_agents.append(int(num))
        elif reading == 'targets':
            T, FN = line.strip().split(' ')
            targets.append(T)
            target_fn.append(FN)

        if re.match(re_cost_function, line):
            reading = 'cost_functions'
        elif re.match(re_edge, line):
            reading = 'edges'
        elif re.match(re_agent, line):
            reading = 'agents'
        elif re.match(re_target, line):
            reading = 'targets'
    f.close()

    agents = np.array(agents)
    targets = np.array(targets)
    edges = np.array(edges)
    vertices = np.unique(edges.flatten())
    agents_idx = np.where(np.isin(vertices, agents))[0].tolist()
    targets_idx = np.where(np.isin(vertices, targets))[0].tolist()

    G = igraph.Graph()
    G.add_vertices(vertices)
    G.add_edges(edges)
    G.es['cost_fn'] = edge_fn
    G.vs[agents_idx]['agent'] = True
    G.vs[agents_idx]['n_agents'] = n_agents
    G.vs[targets_idx]['target'] = True
    G.vs[targets_idx]['value'] = target_fn
    return G, agents, n_agents, targets, functions

def get_agent_routes(G, agent, h, k):
    '''
    Runs BFS to find paths to targets within H hops away.
    '''
    agent_paths = defaultdict(list)
    heap = []

    heappush(heap, [0, agent, [agent]])
    while len(heap):
        hop, node, path = heappop(heap)
        if hop == h:
            break

        neighbors = G.neighbors(node)
        prev_node = None if len(path) == 1 else path[-2]
        for nghbr in neighbors:
            name = G.vs[nghbr]['name']
            if name == prev_node:
                continue

            nghbr_path = path + [name]
            if G.vs[nghbr]['target'] and \
                    len(agent_paths[name]) <= k:
                agent_paths[name].append(nghbr_path)
            
            heappush(heap, [hop+1, name, nghbr_path])
    return agent_paths


if __name__ == '__main__':
    #G = read_graph('env0.txt')
    G = read_graph('env0.txt')

    print(G.vs['target'])
    print(G.vs['agent'])
    print(G.es['cost_fn'])
