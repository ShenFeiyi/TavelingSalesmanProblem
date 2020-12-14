# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from TSP import CityGroup
from Algorithms import brute_force, greedy, NoX, Astar, Genetic

if os.path.exists(os.path.join('.','TSP')):
    imgs = os.listdir(os.path.join('.','TSP'))
    _ = [os.remove(os.path.join('.','TSP',img)) for img in imgs]
else:
    os.mkdir(os.path.join('.','TSP'))

functions = {
    'GREEDY':greedy,
    'FORCE':brute_force,
    'NoX':NoX,
    'A*':Astar,
    'GA':Genetic
    }

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', help='show map')
parser.add_argument('--min', type=int, default=3, help='minimum number of cities in a group (included)')
parser.add_argument('--max', type=int, default=5, help='maximum number of cities in a group (included)')
parser.add_argument('-W', '--width', type=float, default=900, help='width of map (pixels)')
parser.add_argument('-H', '--height', type=float, default=600, help='height of map (pixels)')

params = parser.parse_args()
params = vars(params)

generations = [100,200,400,800,1600,3200]

# time: greedy => 100
# accuracy: greedy => 60
times = [{func:0 for func in functions} for _ in range(params['max']-params['min']+1)]
times_grades = [{func:0 for func in functions} for _ in range(params['max']-params['min']+1)]
length = [{func:0 for func in functions} for _ in range(params['max']-params['min']+1)]
accuracy = [{func:0 for func in functions} for _ in range(params['max']-params['min']+1)]
for iteration in generations:
    for i in range(params['min'],params['max']+1):
        times[i-params['min']]['GA'+str(iteration)] = 0
        times_grades[i-params['min']]['GA'+str(iteration)] = 0
        length[i-params['min']]['GA'+str(iteration)] = 0
        accuracy[i-params['min']]['GA'+str(iteration)] = 0

for _ in range(7):
    with open('log.txt','w',encoding='utf-8') as file:
        for number in range(params['min'],params['max']+1):
            CG = CityGroup(number, params['width'], params['height'])
            file.write('\n'+str(number)+' cities\n\n')
            for func in functions:
                if func == 'GA':
                    for iteration in generations:
                        if iteration > 1000 and number > 10:
                            times[number-params['min']][func+str(iteration)] = np.nan
                            length[number-params['min']][func+str(iteration)] = np.nan
                            continue
                        dt, l = CG.evaluate(functions[func], show=params['show'], iteration=iteration)
                        file.write(f'{func}{iteration}\tdt={dt}\tlength={l}\n')
                        times[number-params['min']][func+str(iteration)] += dt
                        length[number-params['min']][func+str(iteration)] += l
                else:
                    if func == 'FORCE' and number > 10:
                        times[number-params['min']][func] = np.nan
                        length[number-params['min']][func] = np.nan
                        continue
                    dt, l = CG.evaluate(functions[func], show=params['show'])
                    file.write(f'{func}\tdt={dt}\tlength={l}\n')
                    times[number-params['min']][func] += dt
                    length[number-params['min']][func] += l
    print('- '*100)

for index in range(params['max']-params['min']+1):
    times[index]['GA'] = min([times[index]['GA'+str(i)] for i in generations])
    length[index]['GA'] = min([length[index]['GA'+str(i)] for i in generations])

for func in functions:
    for index in range(params['max']-params['min']+1):
        times_grades[index][func] = 100*pow(times[index]['GREEDY']/times[index][func],1/4)
        if index > 10 - params['min']:
            accuracy[index][func] = 60*length[index]['GREEDY']/length[index][func]
        else:
            accuracy[index][func] = 100*length[index]['FORCE']/length[index][func]
for iteration in generations:
    for index in range(params['max']-params['min']+1):
        times_grades[index]['GA'+str(iteration)] = 100*pow(times[index]['GREEDY']/times[index]['GA'+str(iteration)],1/4)
        if index > 10 - params['min']:
            accuracy[index]['GA'+str(iteration)] = 60*length[index]['GREEDY']/length[index]['GA'+str(iteration)]
        else:
            accuracy[index]['GA'+str(iteration)] = 100*length[index]['FORCE']/length[index]['GA'+str(iteration)]

fig = plt.figure(figsize=(18,12), dpi=128)
ax1 = fig.add_subplot(2,1,1)

linestyles = ['-','-.',':','--']
markers = ['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']

X = np.arange(params['min'],params['max']+1)
Y1 = [[] for _ in range(len(times[0]))]
for i, algo in enumerate(times[0]):
    for city in range(params['max']-params['min']+1):
        Y1[i].append(times_grades[city][algo])
Y1 = np.array(Y1)
for i, algo in enumerate(times[0]):
    if algo == 'GREEDY':
        ax1.plot(X, Y1[i], 'k--', label=algo, alpha=0.8)
    elif algo[:2] == 'GA':
        r, g, b = 0, 0xFF, np.random.randint(0xFF)
        color = '#'
        color += str(hex(r))[2:].zfill(2)
        color += str(hex(g))[2:].zfill(2)
        color += str(hex(b))[2:].zfill(2)
        ax1.plot(X, Y1[i], label=algo, color=color, marker=markers[i],
                 linestyle=linestyles[np.random.randint(len(linestyles)-1)])
    else:
        r, g, b = 0xFF, np.random.randint(int(0xFF/2)), 0
        color = '#'
        color += str(hex(r))[2:].zfill(2)
        color += str(hex(g))[2:].zfill(2)
        color += str(hex(b))[2:].zfill(2)
        ax1.plot(X, Y1[i], label=algo, color=color, marker=markers[i],
                 linestyle=linestyles[np.random.randint(len(linestyles)-1)])

ax1.set_title('time grades')
ax1.set_ylabel('grades')

for key, spine in ax1.spines.items():
    # 'left', 'right', 'bottom', 'top'
    if key in ['right', 'bottom', 'top']:
        spine.set_visible(False)

plt.xticks([])
plt.legend(loc='upper right')

ax2 = fig.add_subplot(2,1,2)

X = np.arange(params['min'],params['max']+1)
Y2 = [[] for _ in range(len(times[0]))]
for i, algo in enumerate(times[0]):
    for city in range(params['max']-params['min']+1):
        Y2[i].append(accuracy[city][algo])
Y2 = np.array(Y2)
for i, algo in enumerate(times[0]):
    if algo == 'GREEDY':
        ax2.plot(X, Y2[i], 'k--', label=algo, alpha=0.8)
    elif algo[:2] == 'GA':
        r, g, b = 0, 0xFF, np.random.randint(0xFF)
        color = '#'
        color += str(hex(r))[2:].zfill(2)
        color += str(hex(g))[2:].zfill(2)
        color += str(hex(b))[2:].zfill(2)
        ax2.plot(X, Y2[i], label=algo, color=color, marker=markers[i],
                 linestyle=linestyles[np.random.randint(len(linestyles)-1)])
    else:
        r, g, b = 0xFF, np.random.randint(int(0xFF/2)), 0
        color = '#'
        color += str(hex(r))[2:].zfill(2)
        color += str(hex(g))[2:].zfill(2)
        color += str(hex(b))[2:].zfill(2)
        ax2.plot(X, Y2[i], label=algo, color=color, marker=markers[i],
                 linestyle=linestyles[np.random.randint(len(linestyles)-1)])

ax2.set_title('accuracy grades')
ax2.set_xlabel('cities')
ax2.set_ylabel('grades')

for key, spine in ax2.spines.items():
    # 'left', 'right', 'bottom', 'top'
    if key in ['right', 'top']:
        spine.set_visible(False)

plt.xticks(range(params['min'],params['max']+1), range(params['min'],params['max']+1))
plt.legend(loc='upper right')

plt.savefig('time & accuracy.jpg')
#plt.show()
