'''
we remove the source 
'''

import to_csv

label = {}
source = []
target = []
cascade_root = to_csv.cascade_root
with open('D:\\学习\\博士\\谣言传播仿真\\txt\\twitter_net_30000.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        line = line.replace('\n', '')
        row = line.split(' ')
        source.append(row[0])
        target.append(row[1])
        label[row[0]] = row[2]
        label[row[1]] = row[3]

nodes = list(set(source) | set(target))
reserve = ['1721733', '1095600', '964314', '1008080', '1060703', '667037','2777735', '1975216', '2389312', '1102746', '1003979', '1035865', '984019', '1000886', '1171048', '1866899', '1281243', '1474223']
for vtx in nodes:
    if label[vtx] != '0':
        if cascade_root[vtx] not in reserve:
            label[vtx] = '0'

with open('D:\\学习\\博士\\谣言传播仿真\\txt\\twitter_net_30000_6_12.txt', 'w') as f:
    for i in range(len(source)):
        f.writelines([source[i], ' ', target[i], ' ', label[source[i]], ' ',label[target[i]]])
        f.write('\n')