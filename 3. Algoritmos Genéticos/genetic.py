import tsp as model

def base_algorithm(pop_size, elite_size, max_generations, 
                        crossover_rate=0.9, mutation_rate=0.01):
    population = model.init(pop_size)
    yield 0, population, model.fit(population)
    for g in range(max_generations):
        fitness = model.fit(population)
        elite = model.elitism(population, fitness, elite_size)
        parents = model.selection(population, fitness, pop_size - elite_size)
        children = model.crossover(parents, crossover_rate)
        children = model.mutation(children, mutation_rate)
        population = elite + children
        yield g+1, population, model.fit(population)
        if model.stop(): break