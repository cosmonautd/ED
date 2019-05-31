import numpy
import string
import matplotlib.pyplot as plt

numpy.random.seed(1)

N = 20
CITY_LABELS = list(range(N))
CITY_COORD = numpy.random.randint(0, 200, (N, 2))
CITY_DICT = {label : coord for (label, coord) in zip(CITY_LABELS, CITY_COORD)}

population_history = list()
fitness_history = list()

def init(pop_size):
    def random_permutation():
        population = list()
        for _ in range(pop_size):
            individual = list(numpy.random.permutation(CITY_LABELS))
            population.append(individual)
        return population
    def heuristic():
        raise NotImplementedError
    return random_permutation()

def fit(population):
    fitness = list()
    for individual in population:
        distance = 0
        for i, city in enumerate(individual):
            s = CITY_DICT[individual[i-1]]
            t = CITY_DICT[individual[i]]
            distance += numpy.linalg.norm(s-t)
        fitness.append(1/distance)
    return fitness

def selection(population, fitness, n):
    def roulette():
        idx = numpy.arange(0, len(population))
        probabilities = fitness/numpy.sum(fitness)
        parents_idx = numpy.random.choice(idx, size=n, p=probabilities)
        parents = numpy.take(population, parents_idx, axis=0)
        parents = [(parents[i], parents[i+1]) for i in range(0, len(parents)-1, 2)]
        return parents
    def tournament(K):
        parents = list()
        idx = numpy.arange(0, len(population))
        for _ in range(n):
            turn_idx = numpy.random.choice(idx, size=K)
            turn_fitness = numpy.take(fitness, turn_idx, axis=0)
            argmax = numpy.argmax(turn_fitness)
            parents.append(population[argmax])
        parents = [(parents[i], parents[i+1]) for i in range(0, len(parents)-1, 2)]
    def ranking():
        raise NotImplementedError
    return roulette()

def crossover(parents, crossover_rate=0.9):
    def ordered():
        children = list()
        for pair in parents:
            if numpy.random.random() < crossover_rate:
                for (parent1, parent2) in [(pair[0], pair[1]), (pair[1], pair[0])]:
                    points = numpy.random.randint(0, len(parent1), 2)
                    start = min(points)
                    end   = max(points)
                    segment1 = [x for x in parent1[start:end]]
                    segment2 = [x for x in parent2[end:] if x not in segment1]
                    segment3 = [x for x in parent2[:end] if x not in segment1]
                    child = segment3 + segment1 + segment2
                    children.append(child)
            else:
                children.append(pair[0])
                children.append(pair[1])
        return children
    return ordered()

def mutation(children, mutation_rate=0.01):
    def bitflip():
        raise NotImplementedError
    def swap():
        for i, child in enumerate(children):
            if numpy.random.random() < mutation_rate:
                [a, b] = numpy.random.randint(0, len(child), 2)
                children[i][a], children[i][b] = children[i][b], children[i][a]
        return children
    def inversion():
        raise NotImplementedError
    def scramble():
        raise NotImplementedError
    return swap()

def stop():
    return False

def elitism(population, fitness, n):
    return [e[0] for e in sorted(zip(population, fitness),
                key=lambda x:x[1], reverse=True)[:n]]