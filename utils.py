import re

re_cost_function = r'^#\scost_functions'
re_edges = r'^#\sedges'
re_target = r'^#\stargets'
re_agent = r'^#\sagents'

def read_graph(path):
    reading = ''
    f = open(path, 'r')
    for line in f.readlines():
        if re.match(re_cost_function):
            reading = 'cost_functions'
        elif re.match(re_edge):
            reading = 'edges'
        elif re.match(re_agent):
            reading = 'agents'
        elif re.match(re_target):
            reading = 'targets'
        elif line == '':
            reading = ''
        
        if reading == 'cost_functions':
            pass
    f.close()

