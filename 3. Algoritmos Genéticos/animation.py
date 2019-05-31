import genetic
import tsp as model

import numpy
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

run = genetic.base_algorithm(pop_size=100, elite_size=20, max_generations=300)

matplotlib.rcParams['toolbar'] = 'None'
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))

x = []
y_min = []
y_mean = []
def animate(args):
    ax0.clear()
    ax1.clear()
    ax0.set_xlabel('Gerações')
    ax0.set_ylabel('Distância')
    g, population, fitness = args
    x.append(g)
    dist = [1/f for f in fitness]
    y_min.append(numpy.min(dist))
    y_mean.append(numpy.mean(dist))
    ax0.plot(x, y_min, color='red', alpha=0.7, label='Indivíduo mais apto')
    ax0.plot(x, y_mean, color='blue', alpha=0.7, label='Média da população')
    ax0.legend(loc='upper right')
    ax1.set_title('Indivíduo mais apto')
    ax1.set_xlim([-20, 210])
    ax1.set_ylim([-20, 210])
    ax1.scatter(model.CITY_COORD[:,0], model.CITY_COORD[:,1], color='black', alpha=0.85)
    solution = max(zip(population, fitness), key=lambda x:x[1])[0]
    P = numpy.array([model.CITY_DICT[s] for s in solution] + [model.CITY_DICT[solution[0]]])
    ax1.plot(P[:,0], P[:,1], color='red', alpha=0.85)
    return

anim = matplotlib.animation.FuncAnimation(
        fig, animate, frames=run, interval=10, repeat=False)

plt.tight_layout(pad=3.5)
plt.show()