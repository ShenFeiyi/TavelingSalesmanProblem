# -*- coding:utf-8 -*-
import os
import copy
import argparse
import itertools
import numpy as np
from queue import PriorityQueue
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from TSP import CityGroup, RouteTree, Pigeons

def brute_force(CG):
    """Brute force method for TSP

    NOTE: You can write your own methods like this to solve TSP. This is an example.

    Args:
        CG (CityGroup): A CityGroup object.

    Returns:
        shortest_tour (list): A list of citys. It represents the best order of visiting cities.
    """
    tours = itertools.permutations(CG.cities)
    shortest_tour = min(tours, key=CG.tour_length)
    return shortest_tour

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
        p1 = np.array([c1.x, c1.y])
        p2 = np.array([c2.x, c2.y])
        if p1[0] > p2[0]:
            p1, p2 = p2, p1

        if p1[0] == p2[0]:
            a, b, c = 1, 0, -p1[0]
        else:
            k = (p1[1]-p2[1])/(p1[0]-p2[0])
            a, b = -k, 1
            c = ((-a*p1[0]-b*p1[1])+(-a*p2[0]-b*p2[1]))/2

        return a, b, c, p1[0], p2[0]

    def intersect(line, lines):
        """Calculate the intersection of two lines
        Args:
            line (tuple): aka. (a, b, c, xmin, xmax), where ax+by+c=0 and xmin, xmax are two boundaries.
            lines (list): A list of line.

        Returns:
            X (bool): if two lines intersect with each other, True; else, False. 
        """
        X = False
        a1, b1, c1, min1, max1 = line
        for l in lines:
            a2, b2, c2, min2, max2 = l
            M = np.array([[a1, b1],[a2, b2]])
            C = np.array([[-c1],[-c2]])
            MI = np.linalg.inv(M)
            xy = MI.dot(C)
            x = xy[0]
            if (x >= max(min1,min2)) & (x <= min(max1,max2)):
                X = True
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

if __name__ == '__main__':
    functions = {
        'FORCE':brute_force,
        'GREEDY':greedy,
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
    parser.add_argument('-A', '--algorithm', type=str, default=' '.join([func for func in functions]),
                        help=f"list of algorithm to run, seperate with ' ', e.g. 'FORCE GREEDY',\
                        all the choices are {' '.join([func for func in functions])}.")
    params = parser.parse_args()
    params = vars(params)

    algorithm = params['algorithm'].split(' ')
    params['algorithm'] = [a for a in algorithm]

    with open('log.txt','w',encoding='utf-8') as file:
        for number in range(params['min'],params['max']+1):
            CG = CityGroup(number, params['width'], params['height'])

            for func in params['algorithm']:
                if func == 'GA':
                    for iteration in [100,200,400,800,1600,3200]:
                        dt, l = CG.evaluate(functions[func], show=params['show'], iteration=iteration)
                        file.write(f'{functions[func].__name__}{iteration}\tdt={dt}\tlength={l}\n')
                else:
                    if func == 'FORCE' and number > 8:
                        continue
                    dt, l = CG.evaluate(functions[func], show=params['show'])
                    file.write(f'{functions[func].__name__}\tdt={dt}\tlength={l}\n')
