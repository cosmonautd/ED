import tsp as model

def base_algorithm(pop_size, max_generations, elite_size=0):
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