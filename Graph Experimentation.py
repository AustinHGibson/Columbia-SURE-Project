# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:26:51 2024

@author: austi
"""
import matplotlib.pyplot as plt
ticks = [1,2,3,4]
s = 5
experimental,axes = plt.subplots()

axes.plot([1,2,3,4],[2,3,4,5], color = 'magenta', marker = 'o', label = '1',linestyle='--')
axes.plot([1,2,3,4],[4,6,8,10], linestyle='dotted', marker = 'o', color = 'magenta', label = '1')

axes.set_title("Loss Across Epochs During Task 1 Training (Simulation {:d})".format(s), fontsize = 18)
axes.set_xticks(ticks)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
axes.set_ylabel("Performance", fontsize = 15)
axes.set_xlabel("Simulation #", fontsize = 15)
axes.legend(['Task 1', 'Task 2'], loc = 'upper left', prop = {'size':13})


experimental.savefig('C:/Users/austi/Downloads/Project/Graphs/EXPERIMENTAL.pdf')