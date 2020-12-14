# Travelling Salesman Problem

https://github.com/SDaydreamer/TavelingSalesmanProblem

[TOC]

## 0. Questions

给定一系列城市和每对城市之间的距离，求访问每个城市一次并回到最初起始城市的最短回路，这就是著名的旅行商问题（travelling salesman problem, TSP）。在组合优化中，TSP一个NP困难问题，求解算法在最坏情况下的时间复杂度会随着城市数量增多呈超多项式级别增长。

请设计多种算法求解TSP问题，比较不同算法的优劣性，分析随着问题规模的增大，各算法的求解时间和求解精度的变化。

本题提供暴力枚举法求解TSP问题的Python3代码实现,并提供了一些基础函数。

**需要提交：**

（1）程序文档，文档结构包括：问题描述、主要算法或者模型、实验数据及分析、有关说明（如引用他人程序说明）；

（2）程序源代码，其中需要包含注释，以及程序运行环境的说明；

（3）提交方式：将有关文件打包成 xxP2.zip, 其中xx为学号，并上传到pintia.cn中。



## 1. Requirements 

1. Python 3.7.3
2. Numpy 1.18.5
3. Matplotlib 3.3.2



## 2. Code

### 2.1. TSP.py

#### 2.1.1. `class City`

```python
class City:
    """A class represent city (from code example)

    Attributes:
        x (float): The x coordinate of the city.
        y (float): The y coordinate of the city.
    """
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name
    
    def __repr__(self):
        return "%s:(%d,%d)" % (self.name, self.x, self.y)
```

#### 2.1.2. `class CityGroup`

```python
class CityGroup:
    """A class represent a group of cities

    Attributes:
        n (int): The number of examples to generate.
        width (int, default=900): The range of the x coordinate of the generated cities is [0, width].
        higth (int, default=600): The range of the y coordinate of the generated cities is [0, height].
    """
    def __init__(self, n, width=900, height=600):
        self.width = width
        self.height = height
        self.cities = self.generate_cities(n, width, height)
        self.MAP = plt.figure(figsize=(width/100,height/100), dpi=128)
        self.map = self.MAP.add_subplot(1,1,1)

    def __len__(self):
        return len(self.cities)

    def __repr__(self):
        out = ""
        for city in self.cities:
            out += "%s:(%d,%d)\n" % (city.name, city.x, city.y)
        return out
```

**Methods in `CityGroup`**

**1. `distance`**

```python
def distance(self, A, B):
    """Calculate distance between two cities (from code example)

    Args:
        A (City): A city object.
        B (City): A city object.

    Returns:
        The Euclidean distance between city A and city B. (float)
    """
    return np.sqrt((A.x-B.x)**2 + (A.y-B.y)**2)
```

**2. `generate_city_names`**

```python
def generate_city_names(self, n):
    """Generate example cities' names

    Args:
        n (int): The number of names to generate.

    Returns:
        names (list): A list of generated city names.
    """
    # ...
    return names
```

**3. `generate_cities`**

```python
def generate_cities(self, n, width=900, height=600):
    """Generate example cities (from code example)

    Args:
        n (int): The number of examples to generate.
        width (int, default=900): The range of the x coordinate of the generated cities is [0, width].
        higth (int, default=600): The range of the y coordinate of the generated cities is [0, height].

    Returns:
        cities (set): A set of generated city objects.
    """
    names = self.generate_city_names(n)
    cities = set(City(np.random.rand()*width, np.random.rand()*height, names[i]) for i in range(n))
    return cities
```

**4. `plot_tour`**

```python
def plot_tour(self, tour, dt, algorithm):
    """Plot the tour you take (from code example)

    Args:
        tour (list): A list of city objects. It represents the order of visiting cities.
        dt (float): Total time cost.
        algorithm (str): Algorithm name.

    Returns:
        (None)
    """
    # ...
```

**5. `tour_length`**

```python
def tour_length(self, tour):
    """Calculate total length of the tour you take (from code example)

    Args:
        tour (list): A list of city objects. It represents the order of visiting cities.

    Returns:
        The total length of the tour with the visiting order given. (float)
    """
    return sum(self.distance(tour[i], tour[i-1]) for i in range(1,len(tour)))
```

**6. `evaluate`**

```python
def evaluate(self, algorithm, show=True, **kwarg):
    """Plot the tour of the algorithm and show statistical information (from code example)

    - Validate the tour obtained by the algorithm.
    - Calculate the length of the tour.
    - Calculate the execution time of the algorithm.
    - Plot the tour of the algorithm you take. 

    Args:
        algorithm (function): The algorithm you designed to solve TSP. Refer to `brute_force_tsp()` function.
        show (bool, default=True): True: show & save map; False: save map only

    Returns:
        The time consumption of this algorithm. (float)
        The tour length. (float)
    """
    t0 = time.time()
    try:
        tour = algorithm(self)
    except TypeError:
        # For Genetic Algorithm
        tour = algorithm(self, kwarg['iteration'])
    t1 = time.time()
    # ...
    return t1 - t0, tourLength
```

#### 2.1.3. `class RouteTree`

```python
class RouteTree:
    """A class represent all possible route, for NoX, A*, etc. algorithms. 

    Attributes:
        node (City Object): This node. 
        kids (list): A list of all other City Objects available.
        parent (RouteTree): Parent node.
        children (list): A list of RouteTree Object.
        dead_end (list): A lis of City Objects, may not go that way. 
        distanceTable (dict): A table records all distances between two cities.
                              table = {city0:{city1:d01, city2:d02, ...}, city1: ...}
        g (float): Distance already taken.
        h (float): Estimated distance.
        f (float): g + h, the evaluation function of A* algorithm. 
    """
    def __init__(self, node, kids, distanceTable, **kwarg):
        self.node = node
        self.kids = kids
        self.table = distanceTable
        self.parent = kwarg['parent'] if 'parent' in kwarg else None
        self.dead_end = []

        if self.parent is None:
            self.g = 0
        else:
            self.g = self.parent.g + self.table[self.node][self.parent.node]

    def __repr__(self):
        if self.parent is None:
            return 'Parent: '+str(self.parent)+'\n'\
                   +'Node: '+str(self.node)+'\n'\
                   +'Children: '+str([kid for kid in self.kids if not kid in self.dead_end])
        else:
            return 'Parent: '+str(self.parent.node)+'\n'\
                   +'Node: '+str(self.node)+'\n'\
                   +'Children: '+str([kid for kid in self.kids if not kid in self.dead_end])
```

**Methods & properties in `RouteTree`**

**1. property `f`**

```python
@property
def f(self):
    """Evaluation function in A* algorithm

    Based on this function, the algorithm can decide which node to take as the next step.
    """
    self.h = 0
    for kid in self.kids:
        self.h += self.table[self.node][kid]
    return self.g + self.h
```

**2. property `children`**

```python
@property
def children(self):
    """Generate child nodes
    """
    child_node = []
    for kid in self.kids:
        if not kid in self.dead_end:
            child_node.append(
                RouteTree(kid, [k for k in self.kids if not k is kid], self.table, parent=self)
                )
    return child_node
```

**3. `path`**

```python
def path(self):
    """Final route

    Args:
        self (RouteTree): The current RouteTree object.

    Returns:
        path (list): A list of RouteTree instances. 
    """
    cur, path = self, []
    while cur:
        path.append(cur)
        cur = cur.parent
    return list(reversed(path))
```

**4. `kill`**

```python
def kill(self, city):
    """Function preventing dead end.
    """
    for child in self.children:
        if city == child.node:
            self.dead_end.append(city)
```

#### 2.1.4. `class Pigeon`

```python
class Pigeon:
    """A class represent an individual, for Genetic Algorithm. 

    Attributes:
        cities (list): A list of City Object.
        DNA (numpy.ndarray): An array of number representing different cities,
        					 e.g. np.array([5,6,3,2,7,1,8,4,0]). 
        fitness (float): Individual's fitness to the environment, aka. total length.
        score (float): Fitness score, from 0 to 1. 
        mutation_rate (float, default=0.05): Gene's mutation rate. 
    """
    def __init__(self, cities, distanceTable, **kwarg):
        self.cities = cities
        self.table = distanceTable
        self.mutation_rate = kwarg['mutation'] if 'mutation' in kwarg else 5e-2
        self.DNA = self.encode() if not 'DNA' in kwarg else kwarg['DNA']
        self.score = kwarg['score'] if 'score' in kwarg else 0

    def __repr__(self):
        return 'Gene: '+str(self.DNA)+'\n'+'Fitness: '+str(self.fitness)
```

**Methods & properties in `Pigeon`**

**1. `encode`**

```python
def encode(self):
    """Generate DNA
    """
    gene_pool = np.arange(len(self.cities))
    np.random.shuffle(gene_pool)
    return gene_pool
```

**2. property `fitness`**

```python
@property
def fitness(self):
    """Individual's fitness to the environment, aka. total length.
    """
    f = 0
    for i in range(len(self.DNA)-1):
        index = self.DNA[i]
        next_index = self.DNA[i+1]
        f += self.table[self.cities[index]][self.cities[next_index]]
    return f
```

**3. `mutate`**

```python
def mutate(self, DNA):
    """DNA mutation (swap)
    """
    mutate = np.random.rand()
    mutate_times = np.log(mutate)/np.log(self.mutation_rate)-1
    while mutate_times > 0:
        p1 = np.random.randint(len(DNA))
        p2 = np.random.randint(len(DNA))
        DNA[p1], DNA[p2] = DNA[p2], DNA[p1]
        mutate_times -= 1
    return DNA
```

**4. `duplicate`**

```python
def duplicate(self):
    """Duplicate itself
    """
    dna = self.mutate(self.DNA)
    return Pigeon(self.cities, self.table, DNA=dna, mutation=self.mutation_rate, score=self.score)
```

#### 2.1.5. `class Pigeons`

```python
class Pigeons:
    """A class represent a group of individuals, for Genetic Algorithm. 

    Attributes:
        cities (list): A list of City Object.
        n (int): Population in the group. 
        distanceTable (dict): A table records all distances between two cities.
                              table = {city0:{city1:d01, city2:d02, ...}, city1: ...}
        environment (float, default=0.72): Kill approximately (1-x)% of the population each time.
        iteration (int, default=100): Evolution iterations. 
        mutation_rate (float, default=0.05): Gene's mutation rate. 
        population (list): A list of Pigeon.
        best (list): A list of the best Pigeon(s). 
    """
    def __init__(self, cities, n, distanceTable, **kwarg):
        self.cities = cities
        self.labors = n
        self.table = distanceTable

        self.environment = kwarg['env'] if 'env' in kwarg else 0.72
        self.iteration = kwarg['iteration'] if 'iteration' in kwarg else 100
        self.mutation_rate = kwarg['mutation'] if 'mutation' in kwarg else 5e-2

        self.fitness_max, self.fitness_min = -np.inf, np.inf
        self.score_max, self.score_min = -np.inf, np.inf

        self.population = [Pigeon(self.cities, self.table, mutation=self.mutation_rate) for _ in range(self.labors)]
```

**Methods & properties in `Pigeons`**

**1. `score`**

```python
def score(self):
    """Score all pigeons
    """
    for pigeon in self.population:
        pigeon.score = (self.fitness_max-pigeon.fitness)/(self.fitness_max-self.fitness_min)
```

**2. `overall`**

```python
def overall(self):
    """Determine the maximum and minimum of fitness and score
    """
    self.fitness_max, self.fitness_min = -np.inf, np.inf
    self.score_max, self.score_min = -np.inf, np.inf
    for pigeon in self.population:
        if pigeon.fitness > self.fitness_max:
            self.fitness_max = pigeon.fitness
        if pigeon.fitness < self.fitness_min:
            self.fitness_min = pigeon.fitness
        if pigeon.score > self.score_max:
            self.score_max = pigeon.score
        if pigeon.score < self.score_min:
            self.score_min = pigeon.score
```

**3. `kill`**

```python
def kill(self):
    """Kill pigeons with low score
    """
    for pigeon in self.population:
        if np.random.rand()*self.environment > pigeon.score:
            self.population.pop(self.population.index(pigeon))
```

**4. `duplicate`**

```python
def duplicate(self):
    """Duplicate pigeons
    """
    nlack = self.labors - len(self.population)
    while nlack > 0:
        pigeon = self.population[np.random.randint(len(self.population))].duplicate()
        self.population.append(pigeon)
        nlack -= 1
```

**5. `evolution`**

```python
def evolution(self):
    """Pigeon evolution
    """
    epochs = self.iteration
    while epochs > 0:
        self.score()
        self.overall()
        self.kill()
        self.duplicate()
        epochs -= 1
```

**6. property `best`**

```python
@property
def best(self):
    """The best pigeon
    """
    self.overall()
    for pigeon in self.population:
        if pigeon.fitness == self.fitness_min:
            return pigeon
    return None
```

### 2.2. Algorithms.py

#### 2.2.1. `brute_force`

```python
def brute_force(CG):
    """Brute force method for TSP (from code example)

    NOTE: You can write your own methods like this to solve TSP. This is an example.

    Args:
        CG (CityGroup): A CityGroup object.

    Returns:
        shortest_tour (list): A list of citys. It represents the best order of visiting cities.
    """
    tours = itertools.permutations(CG.cities)
    shortest_tour = min(tours, key=CG.tour_length)
    return shortest_tour
```

#### 2.2.2. `greedy`

```python
def greedy(CG):
    """Greedy method for TSP

    Args:
        CG (CityGroup): A CityGroup object.

    Returns:
        shortest_tour (list): A list of citys. It represents the best order of visiting cities.
    """
    cities = list(CG.cities) # [City0, City1, ...]
    distanceTable = {}
    for i in range(len(CG)):
        distanceTable[cities[i]] = {}
        for j in range(len(CG)):
            distanceTable[cities[i]][cities[j]] = CG.distance(cities[i], cities[j])

    tour_order = [[] for _ in range(len(CG))] # [[City0, CityA, CityB, ...], [City1, CityA, CityB, ...], ...]
    distance = np.zeros((len(CG)),dtype='float64') # total distance [xxx, xxx, ...]
    for icity, city in enumerate(cities):
        tour = cities.copy()
        tour_order[icity].append(tour.pop(icity)) # start
        while len(tour) > 0:
            d = [distanceTable[city][c] for c in tour]
            tour_order[icity].append(tour.pop(d.index(min(d))))
            distance[icity] += min(d)

    dis_min_index = np.argmin(distance)
    shortest_tour = tour_order[dis_min_index]
    return shortest_tour
```

#### 2.2.3. `NoX`

```python
def NoX(CG):
    """No Cross Over method for TSP (based on greedy method)
    Args:
        CG (CityGroup): A CityGroup object.

    Returns:
        shortest_tour (list): A list of citys. It represents the best order of visiting cities.
    """
    def line(c1, c2):
        """Calculate a line equation
        Args:
            c1, c2 (City): Two City Objects.

        Returns:
            line (tuple): aka. (a, b, c, xmin, xmax), where ax+by+c=0 and xmin, xmax are two boundaries. 
        """
        # ...
        return a, b, c, p1[0], p2[0]

    def intersect(line, lines):
        """Calculate the intersection of two lines
        Args:
            line (tuple): aka. (a, b, c, xmin, xmax), where ax+by+c=0 and xmin, xmax are two boundaries.
            lines (list): A list of line.

        Returns:
            X (bool): if two lines intersect with each other, True; else, False. 
        """
        # ...
        return X

    cities = list(CG.cities) # [City0, City1, ...]
    distanceTable = {}
    for i in range(len(CG)):
        distanceTable[cities[i]] = {}
        for j in range(len(CG)):
            distanceTable[cities[i]][cities[j]] = CG.distance(cities[i], cities[j])

    totalDistance = []
    paths = []
    for start in cities:
        kids = [city for city in cities if not city is start]
        root = RouteTree(start, kids, distanceTable)
        while len(root.children) > 0:
            # draw previous lines
            p = copy.deepcopy(root)
            lines = []
            while p.parent:
                lines.append(line(p.node, p.parent.node))
                p = p.parent
            # calculate distance to all cities available
            distance = []
            for child in root.children:
                distance.append(distanceTable[child.node][root.node])
            # the shortest distance
            jack = root.children[distance.index(min(distance))]
            newline = line(jack.node, root.node)
            try:
                # if intersect with previous lines
                # change to the second/third shortest path
                while intersect(newline, lines):
                    distance = []
                    root.kill(jack.node)
                    for child in root.children:
                        distance.append(distanceTable[child.node][root.node])
                    try:
                        jack = root.children[distance.index(min(distance))]
                    except ValueError:
                        raise IndexError
                    newline = line(jack.node, root.node)
            except IndexError:
                # root.children is empty
                dead_end = copy.deepcopy(root)
                root = root.parent
                root.kill(dead_end.node)

            root = jack
        # end of `while len(root.children) > 0:`
        paths.append([p.node for p in root.path()])
        d = 0
        for i in range(len(paths[-1])-1):
            c1 = paths[-1][i]
            c2 = paths[-1][i+1]
            d += distanceTable[c1][c2]
        totalDistance.append(d)
    # bugs!!! may still cross over when return to start or in other situations
    return paths[totalDistance.index(min(totalDistance))]
```

#### 2.2.4. `Astar`

```python
def Astar(CG):
    """A* method for TSP
    Args:
        CG (CityGroup): A CityGroup object.

    Returns:
        shortest_tour (list): A list of citys. It represents the best order of visiting cities.
    """
    cities = list(CG.cities)
    distanceTable = {}
    for i in range(len(CG)):
        distanceTable[cities[i]] = {}
        for j in range(len(CG)):
            distanceTable[cities[i]][cities[j]] = CG.distance(cities[i], cities[j])

    path = []
    q = PriorityQueue()
    for start in cities:
        kids = [city for city in cities if not city is start]
        root = RouteTree(start, kids, distanceTable)
        while len(root.children) > 0:
            for child in root.children:
                q.put((child.f, child.h, child.g, child))
            king = q.get()
            root = king[3]
        path.append(root.path())

    distances = []
    for p in path:
        d = 0
        for i in range(len(p)-1):
            d += distanceTable[p[i].node][p[i+1].node]
        distances.append(d)

    shortest_path = path[distances.index(min(distances))]
    shortest_tour = [ path.node for path in shortest_path ]
    return shortest_tour
```

#### 2.2.5. `Genetic`

```python
def Genetic(CG, iteration):
    """Genetic algorithm method for TSP
    Args:
        CG (CityGroup): A CityGroup object.

    Returns:
        shortest_tour (list): A list of citys. It represents the best order of visiting cities.
    """
    def factorial(n):
        return n*factorial(n-1) if n > 1 else 1

    cities = list(CG.cities)
    distanceTable = {}
    for i in range(len(CG)):
        distanceTable[cities[i]] = {}
        for j in range(len(CG)):
            distanceTable[cities[i]][cities[j]] = CG.distance(cities[i], cities[j])

    labors = factorial(len(cities)) if factorial(len(cities)) < 1000 else 1000
    pigeons = Pigeons(cities, labors, distanceTable, iteration=iteration)
    pigeons.evolution()
    best = pigeons.best

    return [ cities[i] for i in best.DNA ]
```

### 2.3. Solver.py

##### Illustrate results. 

###### # Import packages

```python
# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from TSP import CityGroup
from Algorithms import brute_force, greedy, NoX, Astar, Genetic
```

###### # Empty folder

```python
if os.path.exists(os.path.join('.','TSP')):
    imgs = os.listdir(os.path.join('.','TSP'))
    _ = [os.remove(os.path.join('.','TSP',img)) for img in imgs]
else:
    os.mkdir(os.path.join('.','TSP'))
```

###### # All algprithms

```python
functions = {
    'GREEDY':greedy,
    'FORCE':brute_force,
    'NoX':NoX,
    'A*':Astar,
    'GA':Genetic
    }
```

###### # Parser

```python
parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', help='show map')
parser.add_argument('--min', type=int, default=3, help='minimum number of cities in a group (included)')
parser.add_argument('--max', type=int, default=5, help='maximum number of cities in a group (included)')
parser.add_argument('-W', '--width', type=float, default=900, help='width of map (pixels)')
parser.add_argument('-H', '--height', type=float, default=600, help='height of map (pixels)')

params = parser.parse_args()
params = vars(params)
```

###### # Initialization

```python
generations = [100,200,400,800,1600,3200]
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
```

###### # Calculate results

```python
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
```

###### # Grading

```python
# time: greedy => 100
# accuracy: greedy => 60
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

```

###### # Plot data

```python
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
```



## 3. Results

<img src='time & accuracy -2.jpg' style='zoom:50%'>

<img src='time & accuracy -0.jpg' style='zoom:50%'>

These two results share something in common. 

When number of cities greater than 10, 

Time: $Greedy>A*>NoX>GeneticAlgotirhm>brute\_force$

Accuracy: $brute\_force>NoX\approx A*>Greedy>GeneticAlgprithm$



## 4. Conclusion

If there's a better way to estimate `h`, `A*` algorithm could do better. 

If there's a better way for genetic algorithm to iterate, it could do better. 