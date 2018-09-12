'''
we add inactive nodes to the cascade mode acquired from the 'to_csv.py' in proportion
'''

import to_csv
import igraph
import random
import numpy as np
import pandas as pd

nodes = to_csv.nodes
category = to_csv.category
state = []
for i in category:
        if i == 'TRUE':
            state.append(1)
        else:
            state.append(2)

edges = []
for i in range(len(to_csv.source)):
    edges.append((to_csv.source[i], to_csv.target[i]))

num = int(len(nodes)/3)
for i in range(num):
    nodes.append(str(i))
    state.append(0)

g = igraph.Graph(directed = 'TRUE')
g.add_vertices(nodes)
g.add_edges(edges)

adj = g.get_adjlist(mode = 'OUT')
degree2num = {}
for i in adj:
    if len(i) not in degree2num.keys():
        degree2num[len(i)] = 1
    else:
        degree2num[len(i)] += 1

### caculate max degree
degree = []
for key, value in degree2num.items():
    degree.append(key)
max_degree = max(degree)

### build friend network
source = to_csv.source
target = to_csv.target
for i in range(len(nodes)):
    num = len(adj[i])
    new_edge_num = int(max_degree/degree2num[len(adj[i])])
    if new_edge_num > num:
        store = random.sample(list(set(nodes)-set(adj[i])),new_edge_num-num)
        for j in store:
            source.append(nodes[i])
            target.append(j)

if __name__ == '__main__':
    with open('D:\\学习\\博士\\谣言传播仿真\\txt\\twitter_net_30000_25.txt', 'a') as f:
        for i in range(len(source)):
            f.writelines([str(source[i]), ' ', str(target[i]), ' ', str(state[nodes.index(source[i])]), ' ', str(state[nodes.index(target[i])]), '\n'])
