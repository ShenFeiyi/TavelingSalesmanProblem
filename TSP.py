# -*- coding:utf-8 -*-
import os
import time
import argparse
import itertools
import numpy as np
from queue import PriorityQueue
from matplotlib import pyplot as plt

class City:
    """A class represent city

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

    def distance(self, A, B):
        """Calculate distance between two cities

        Args:
            A (City): A city object.
            B (City): A city object.
        
        Returns:
            The Euclidean distance between city A and city B. (float)
        """
        return np.sqrt((A.x-B.x)**2 + (A.y-B.y)**2)

    def generate_city_names(self, n):
        """Generate example cities' names

        Args:
            n (int): The number of names to generate.

        Returns:
            names (list): A list of generated city names.
        """
        cities = []
        with open('cities_in_China.txt','r') as file:
            content = file.readlines()

        for c in content:
            cities.append(c[:-1])

        numbers = []
        while len(set(numbers)) < n:
            numbers = [np.random.randint(len(cities)) for i in range(n)]

        names = [cities[i] for i in numbers]
        return names

    def generate_cities(self, n, width=900, height=600):
        """Generate example cities

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

    def plot_tour(self, tour, dt, algorithm):
        """Plot the tour you take

        Args:
            tour (list): A list of city objects. It represents the order of visiting cities.
            dt (float): Total time cost.
            algorithm (str): Algorithm name.

        Returns:
            (None)
        """
        def plot_line(points, style='bo-'):
            plt.rcParams['font.sans-serif'] = ['Hannotate SC'] # display Chinese
            plt.plot([p.x for p in points], [p.y for p in points], style)
            for i in range(len(points)-1):
                text = points[i].name
                text += '\n(%d,%d)' % (points[i].x,points[i].y)
                plt.text(points[i].x, points[i].y, text, fontsize=12)
                self.map.annotate('',[points[i+1].x,points[i+1].y],[points[i].x,points[i].y],
                                  arrowprops={'arrowstyle':'->','color':'b','shrinkA':15,'shrinkB':25})
                self.map.annotate('',[points[i].x,points[i].y],[points[i+1].x,points[i+1].y],
                                  arrowprops={'arrowstyle':'<-','color':'b','shrinkA':15,'shrinkB':25})
            plt.axis('scaled')
            plt.axis('off')

        plot_line(list(tour) + [tour[0]])
        plot_line([tour[0]], 'rs')
        plt.xlim(0,self.width)
        plt.ylim(0,self.height)
        plt.title("{} city tour with length {:.1f} in {:.3f} seconds for {}"
                  .format(len(tour), self.tour_length(tour), dt, algorithm))

    def tour_length(self, tour):
        """Calculate total length of the tour you take 

        Args:
            tour (list): A list of city objects. It represents the order of visiting cities.

        Returns:
            The total length of the tour with the visiting order given. (float)
        """
        return sum(self.distance(tour[i], tour[i-1]) for i in range(1,len(tour)))

    def evaluate(self, algorithm, show=True, **kwarg):
        """Plot the tour of the algorithm and show statistical information

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

        def valid_tour(tour, cities):
            return set(tour) == set(cities) and len(tour) == len(cities)
        assert valid_tour(tour, self.cities)

        a = algorithm.__name__ if not 'iteration' in kwarg else algorithm.__name__+' generations='+str(kwarg['iteration'])
        self.plot_tour(tour, t1-t0, a)
        plt.savefig(os.path.join('.','TSP',str(len(tour))+'-['+a+'].png'))
        if show:
            plt.show()
        plt.clf()
        self.map = self.MAP.add_subplot(1,1,1)
        tourLength = self.tour_length(tour)
        print("{}, {} city tour with length {:.1f} in {:.3f} seconds for {}.\n"
              .format(tour, len(tour), tourLength, t1 - t0, a))
        return t1 - t0, tourLength

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

    @property
    def f(self):
        """Evaluation function in A* algorithm

        Based on this function, the algorithm can decide which node to take as the next step.
        """
        self.h = 0
        for kid in self.kids:
            self.h += self.table[self.node][kid]
        return self.g + self.h

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

    def __repr__(self):
        if self.parent is None:
            return 'Parent: '+str(self.parent)+'\n'\
                   +'Node: '+str(self.node)+'\n'\
                   +'Children: '+str([kid for kid in self.kids if not kid in self.dead_end])
        else:
            return 'Parent: '+str(self.parent.node)+'\n'\
                   +'Node: '+str(self.node)+'\n'\
                   +'Children: '+str([kid for kid in self.kids if not kid in self.dead_end])

    def kill(self, city):
        """Function preventing dead end.
        """
        for child in self.children:
            if city == child.node:
                self.dead_end.append(city)

class Pigeon:
    """A class represent an individual, for Genetic Algorithm. 

    Attributes:
        cities (list): A list of City Object.
        DNA (numpy.ndarray): An array of number representing different cities, e.g. np.array([5,6,3,2,7,1,8,4,0]). 
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

    def encode(self):
        gene_pool = np.arange(len(self.cities))
        np.random.shuffle(gene_pool)
        return gene_pool

    @property
    def fitness(self):
        f = 0
        for i in range(len(self.DNA)-1):
            index = self.DNA[i]
            next_index = self.DNA[i+1]
            f += self.table[self.cities[index]][self.cities[next_index]]
        return f

    def mutate(self, DNA):
        mutate = np.random.rand()
        mutate_times = np.log(mutate)/np.log(self.mutation_rate)-1
        while mutate_times > 0:
            p1 = np.random.randint(len(DNA))
            p2 = np.random.randint(len(DNA))
            DNA[p1], DNA[p2] = DNA[p2], DNA[p1]
            mutate_times -= 1
        return DNA

    def duplicate(self):
        dna = self.mutate(self.DNA)
        return Pigeon(self.cities, self.table, DNA=dna, mutation=self.mutation_rate, score=self.score)

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

    def score(self):
        for pigeon in self.population:
            pigeon.score = (self.fitness_max-pigeon.fitness)/(self.fitness_max-self.fitness_min)

    def overall(self):
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

    def kill(self):
        for pigeon in self.population:
            if np.random.rand()*self.environment > pigeon.score:
                self.population.pop(self.population.index(pigeon))

    def duplicate(self):
        nlack = self.labors - len(self.population)
        while nlack > 0:
            pigeon = self.population[np.random.randint(len(self.population))].duplicate()
            self.population.append(pigeon)
            nlack -= 1

    def evolution(self):
        epochs = self.iteration
        while epochs > 0:
            self.score()
            self.overall()
            self.kill()
            self.duplicate()
            epochs -= 1

    @property
    def best(self):
        self.overall()
        for pigeon in self.population:
            if pigeon.fitness == self.fitness_min:
                return pigeon
        return None

def brute_force_tsp(CG):
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

def greedy_tsp(CG):
    """Greedy method for TSP

    Args:
        CG (CityGroup): A CityGroup object.

    Returns:
        shortest_tour (list): A list of citys. It represents the best order of visiting cities.
    """
    cities = list(CG.cities) # [City0, City1, ...]
    tour_order = [[] for _ in range(len(CG))] # [[City0, CityA, CityB, ...], [City1, CityA, CityB, ...], ...]
    distance = np.zeros((len(CG)),dtype='float64') # total distance [xxx, xxx, ...]
    for icity, city in enumerate(cities):
        tour = cities.copy()
        tour_order[icity].append(tour.pop(icity))
        while len(tour) > 0:
            d = [CG.distance(city,c) for c in tour]
            tour_order[icity].append(tour.pop(d.index(min(d))))
            distance[icity] += min(d)

    dis_min_index = np.argmin(distance)
    shortest_tour = tour_order[dis_min_index]
    return shortest_tour

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', help='show map')
    parser.add_argument('-N', '--number', type=int, default=5, help='number of cities in a group')
    parser.add_argument('-W', '--width', type=float, default=900, help='width of map (pixels)')
    parser.add_argument('-H', '--height', type=float, default=600, help='height of map (pixels)')
    params = parser.parse_args()
    params = vars(params)

    CG = CityGroup(params['number'], params['width'], params['height'])
    dt0, l1 = CG.evaluate(brute_force_tsp, show=params['show'])
    dt1, l2 = CG.evaluate(greedy_tsp, show=params['show'])
    print(dt0,l1,dt1,l2)
