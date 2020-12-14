import math
import random
import time
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class City:
    """A class represent city

    Attributes:
        x (float): The x coordinate of the city.
        y (float): The y coordinate of the city.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return "(%d,%d)"%(self.x, self.y)

def distance(A, B): 
    """Calculate distance between two cities

    Args:
        A (City): A city object.
        B (City): A city object.
    
    Returns:
        The Euclidean distance between city A and city B. (float)
    """
    return math.sqrt((A.x-B.x)**2 + (A.y-B.y)**2)

def gen_cities(n, width=900, height=600, seed=100):
    """Generate example cities

    Args:
        n (int): The number of examples to generate.
        width (int, default=900): The range of the x coordinate of the generated cities is [0, width].
        higth (int, default=600): The range of the y coordinate of the generated cities is [0, height].
        seed (int, default=100): The random seed of the generator.

    Returns:
        cities (set): A set of generated city objects.
    """
    random.seed(seed * n)
    cities = set(City(random.randrange(width), random.randrange(height)) for _ in range(n))
    return cities

def plot_tour(tour):
    """Plot the tour you take

    Args:
        tour (list): A list of city objects. It represents the order of visiting cities.

    Returns:
        (None)
    """
    def plot_lines(points, style='bo-'):
        plt.plot([p.x for p in points], [p.y for p in points], style)
        plt.axis('scaled'); plt.axis('off')

    plot_lines(list(tour) + [tour[0]])
    plot_lines([tour[0]], 'rs')
    

def brute_force_tsp(cities):
    """Brute force method for TSP

    NOTE: You can write your own methods like this to solve TSP. This is an example.

    Args:
        cities (set): A set of citys you generated with function `gen_cities()`.

    Returns:
        shortest_tour (list): A list of citys. It represents the best order of visiting cities.
    """
    tours = itertools.permutations(cities)
    shortest_tour = min(tours, key=tour_length)
    return shortest_tour

def tour_length(tour):
    """Calculate total length of the tour you take 

    Args:
        tour (list): A list of city objects. It represents the order of visiting cities.

    Returns:
        The total length of the tour with the visiting order given. (float)
    """
    return sum(distance(tour[i], tour[i-1]) for i in range(len(tour)))


def evaluate(algorithm, cities):
    """Plot the tour of the algorithm and show statistical information

    - Validate the tour obtained by the algorithm.
    - Calculate the length of the tour.
    - Calculate the execution time of the algorithm.
    - Plot the tour of the algorithm you take. 

    Args:
        algorithm (function): The algorithm you designed to solve TSP. Refer to `brute_force_tsp()` function.
        cities (set): A set of citys you generated with function `gen_cities()`.

    Returns:
        (None)
    """
    t0 = time.clock()
    tour = algorithm(cities)
    t1 = time.clock()

    def valid_tour(tour, cities):
        return set(tour) == set(cities) and len(tour) == len(cities)
    assert valid_tour(tour, cities)

    plot_tour(tour); plt.show()
    print("{} city tour with length {:.1f} in {:.3f} seconds for {}"
          .format(len(tour), tour_length(tour), t1 - t0, algorithm.__name__))
    


if __name__ == "__main__":
    # an example to show how to use these functions
    # generate 8 cities and use brute force search method to solve TSP
    evaluate(brute_force_tsp, gen_cities(8))
