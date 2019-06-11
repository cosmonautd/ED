import knapsack as model

import numpy
import matplotlib
import matplotlib.pyplot as plt

# Algoritmo genético base
# Os detalhes de modelagem do problema são abstraídos no módulo model
def base_algorithm(pop_size, max_generations, elite_size=0):
    population = model.init(pop_size)
    yield 0, population, model.fit(population)
    for g in range(max_generations):
        fitness = model.fit(population)
        elite = model.elitism(population, fitness, elite_size)
        parents = model.selection(population, fitness, pop_size - elite_size)
        children = model.crossover(parents, 0.9)
        children = model.mutation(children, 0.1)
        population = elite + children
        yield g+1, population, model.fit(population)
        if model.stop(): break

# Definição dos parâmetros do algoritmo genético e execução
gen = base_algorithm(pop_size=300, max_generations=200, elite_size=10)
gen = list(gen)

# Obtenção das listas completas de gerações executadas, populações e valores de fitness
g =   [x[0] for x in gen]
pop = [x[1] for x in gen]
fit = [x[2] for x in gen]

# Obtenção da melhor solução presente na última geração
solution = max(zip(pop[-1], fit[-1]), key=lambda x:x[1])[0]

# Cálculo do lucro e do peso da melhor solução encontrada
profit = numpy.sum(numpy.array(solution)*numpy.array(model.p))
weight = numpy.sum(numpy.array(solution)*numpy.array(model.w))

# Cálculo do lucro máximo
max_profit = model.maxknapsack()

# Impressão dos valores obtidos
print('Max Profit: %d' % max_profit)
print('Solution', solution)
print('Profit: %d' % profit)
print('Weight: %d' % weight)

# Plot de gráfico exibindo o comportamento das aptidões no decorrer das gerações
matplotlib.rcParams['toolbar'] = 'None'
fig, (ax0) = plt.subplots(ncols=1, figsize=(10, 5))
ax0.set_xlabel('Gerações')
ax0.set_ylabel('Aptidão') 
y_max  = [numpy.max(p) for p in fit]
y_mean = [numpy.mean(p) for p in fit]
ax0.plot(g, len(g)*[max_profit], "r--", alpha=0.7, label="Referência (m)")
ax0.plot(g, y_max, color='red', alpha=0.7, label='Indivíduo mais apto')
ax0.plot(g, y_mean, color='blue', alpha=0.7, label='Média da população')
ax0.legend(loc='lower right')

plt.show()
