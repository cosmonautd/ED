import tsp as model
import numpy

def debug(var):
    print(var)
    quit()

def ga(pop_size, elite_size, max_generations):
    population = model.init(pop_size)
    yield 0, population, model.fit(population)
    for g in range(max_generations):
        fitness = model.fit(population)
        elite = model.elitism(population, fitness, elite_size)
        parents = model.selection(population, fitness, pop_size - elite_size)
        children = model.crossover(parents)
        children = model.mutation(children)
        population = elite + children
        yield g+1, population, model.fit(population)
        if model.stop(): break

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

matplotlib.rcParams['toolbar'] = 'None'

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))

x = []
y = []
def animate(args):
    ax0.clear()
    ax1.clear()
    ax0.set_xlabel('Gerações')
    ax0.set_ylabel('Distância média das soluções')
    g, population, fitness = args
    x.append(g)
    y.append(numpy.mean([1/f for f in fitness]))
    ax0.plot(x, y, color='blue', alpha=0.5)
    ax1.scatter(model.CITY_COORD[:,0], model.CITY_COORD[:,1], alpha=0.5)
    solution = max(zip(population, fitness), key=lambda x:x[1])[0]
    P = numpy.array([model.CITY_DICT[s] for s in solution] + [model.CITY_DICT[solution[0]]])
    ax1.plot(P[:,0], P[:,1], color='red', alpha=0.85)
    return

anim = matplotlib.animation.FuncAnimation(fig, animate, frames=ga(100, 20, 400), interval=10, repeat=False)
plt.tight_layout(pad=3.5)
plt.show()