import matplotlib.pyplot as plt
import numpy as np

x = [1, 16, 31, 46]
y1 = [0.94,	0.94, 0.94, 0.95]
y2 = [0.70, 0.65, 0.62, 0.71]
y3 = [0.69, 0.69, 0.61, 0.70]
y4 = [0.90, 0.90, 0.88, 0.87]

plt.figure(figsize=(16,9))
plt.plot(x, y1, color='black', linewidth=5, linestyle='-.', marker='o', markeredgewidth=5, markerfacecolor='none', markersize=30, label='TRSD')
plt.plot(x, y2, color='black', linewidth=5, linestyle=':', marker='^', markeredgewidth=5, markerfacecolor='none', markersize=30, label='Greedy')
plt.plot(x, y3, color='black', linewidth=4.5, linestyle='-',  marker='D', markeredgewidth=5, markerfacecolor='none', markersize=30, label='Max-Deg')
plt.plot(x, y4, color='black', linewidth=5, linestyle='--',  marker='s', markeredgewidth=5, markerfacecolor='none', markersize=30, label='SISI')
plt.legend(loc='lower center', fontsize=32, ncol=4)
# set the range of x and y
plt.xlim((0, 47))
plt.ylim((0.5,1))
# set the x and y
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
# set the ticks of x and y
plt.xticks([1, 16, 31, 46], [1, 5, 25, 50], size=35)
yticks = np.linspace(0.5, 1, 6)
plt.yticks(yticks, size=35)
#set the label of x and y
plt.xlabel('Number of rumor source nodes', size=50)
plt.ylabel('Matching(%)', size=55)
# show grid
plt.grid(linestyle='--')
# show
plt.show()