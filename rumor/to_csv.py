'''
according to the .csv file acquired from the 'simplify_csv.py', we make a cascade mode 
include rumor nodes and truth nodes. 
'''

import csv
import numpy as np
import pandas as pd

source = []
target = []
tw_id = []
fake = {}
rumor_id = {}
parent_id = {}
rumor2tid = {}
cascade_root = {}
with open('D:\\学习\\博士\\谣言传播仿真\\twitter_6_12.csv') as f:
    data = csv.reader(f)
    rows = []
    for row in data:
        if data.line_num == 1:
            continue
        if row[3] == 'TRUE' or row[3] == 'FALSE':
            tw_id.append(row[0])
            parent_id[row[0]] = row[2]
            fake[row[0]] = row[3]
            rumor_id[row[0]] = row[1]
            cascade_root[row[0]] = row[4]

for key, value in rumor_id.items():
    if value not in rumor2tid.keys():
        rumor2tid[value] = []
        rumor2tid[value].append(key)
    else:
        rumor2tid[value].append(key)            

category = []
for vertex in tw_id:
    if parent_id[vertex] != '-1':
        source.append(parent_id[vertex])
        target.append(vertex)

nodes = list(set(source) | set(target))
for vertex in nodes:
        fake[vertex] = fake[cascade_root[vertex]]

for vertex in nodes:
    category.append(fake[vertex])

if __name__ == '__main__':
    name = ['id', 'label', 'category']
    data = [nodes, nodes, category]
    new = np.array(data)
    new = list(new.T)
    test = pd.DataFrame(columns = name, data = new)
    test.to_csv('D:\\学习\\博士\\谣言传播仿真\\nodes_1.csv')

    name = ['source', 'target']
    data = [source, target]
    new = np.array(data)
    new = list(new.T)
    test = pd.DataFrame(columns = name, data = new)
    test.to_csv('D:\\学习\\博士\\谣言传播仿真\\edges_1.csv')