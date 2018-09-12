from datetime import datetime
import numpy as np

print(datetime.now())
Dir = 'D://学习//毕业设计//参考文献//同名区分//vector.txt'
vector = {}
cnt = 0
with open(Dir) as file:
    for line in file:
        cnt = cnt+1
        if cnt <= 10:
            continue
        line = line.split()
        tmp = []
        for i in range(1, len(line)):
            tmp.append(float(line[i]))
        vector[line[0]] = tmp
print(datetime.now())
print(np.dot(vector['community'], vector['detect']))
print(datetime.now())