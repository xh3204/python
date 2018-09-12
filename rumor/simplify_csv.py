'''
extract the edges from the file named 'raw_data_anon.csv',and output the new .csv file.
'''

import numpy as np
import csv
import pandas as pd
from datetime import datetime

print(datetime.now())
print('loading...')
source = ['1721733', '1095600', '964314', '1008080', '1060703', '667037', '2777735', '1975216', '2389312', '1102746', '1003979', '1035865', '984019', '1000886', '1171048', '1866899', '1281243', '1474223']
with open('D:\\学习\\博士\\数据集\\falsenews\\FalseNews_Code_Data\\data\\raw_data_anon.csv') as f:
    data = csv.reader(f)
    rows = []
    for row in data:
        if data.line_num == 1:
            continue
        date = row[6].split('-')
        if row[12] in source:
            rows.append([row[0], row[3], row[5], row[8], row[12]])

name = ['tw_id', 'rumor_id', 'parent_id', 'fake', 'cascade_root']
test = pd.DataFrame(columns = name, data = rows)
test.to_csv('D:\\学习\\博士\\谣言传播仿真\\twitter_6_12.csv')
print(datetime.now())