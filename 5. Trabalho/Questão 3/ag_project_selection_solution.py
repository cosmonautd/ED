import numpy
import matplotlib
import matplotlib.pyplot as plt

# Import das definições do modelo de algoritmo genético
import project_selection_solution as model

# Impressão das condições do problema
# Verba disponível
# Quantida de projetos submetidos
# Custos de implementação de cada projeto
# Estimativas de quantidade de pessoas beneficiadas por cada projeto
print('Verba: %dM' % (model.V))
print('Projetos submetidos: %d' % (model.N))
print('Custos dos projetos:')
print(model.c)
print('Estimativas de pessoas beneficiadas:')
print(model.q)

# Algoritmo genético
# Os detalhes de modelagem do problema são abstraídos pelo import de model
def genetic_algorithm(pop_size, max_generations, crossover_rate, mutation_rate, elite_size=0):
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

# Definição dos parâmetros do algoritmo genético e execução
# TODO: Ajustar os hiperparâmetros do algoritmo genético
gen = genetic_algorithm(pop_size=500, max_generations=100, 
                        crossover_rate=0.9, mutation_rate=0.2, elite_size=10)
gen = list(gen)

# Obtenção das listas completas de gerações executadas, populações e valores de fitness
g =   [x[0] for x in gen]
pop = [x[1] for x in gen]
fit = [x[2] for x in gen]

# Obtenção da melhor solução presente na última geração
solution = max(zip(pop[-1], fit[-1]), key=lambda x:x[1])[0]

# Quantidades totais de pessoas beneficiadas e custo dos projetos selecionados
Q = numpy.sum(numpy.array(solution)*numpy.array(model.q))
C = numpy.sum(numpy.array(solution)*numpy.array(model.c))

# Cálculo da quantidade total de pessoas beneficiadas pela melhor solução possível
max_Q = model.maxknapsack()

# Impressão dos valores obtidos pelo algoritmo genético
print()
print('Projetos selecionados', solution)
print('Custo total: %dM' % C)
print('Total de pessoas beneficiadas: %dM' % Q)
print('Total de pessoas beneficiadas na melhor solução possível: %dM' % max_Q)

# Se o custo da solução encontrada for superior à verba, algo está errado...
if C > model.V:
    print()
    print('Erro: Custo dos projetos selecionados é superior à verba')

# Plot de gráficos exibindo comportamento de variáveis de interesse no decorrer das gerações
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
ax0.set_xlabel('Gerações')
ax0.set_ylabel('Total de pessoas beneficiadas (M)')
y_max  = [numpy.sum(numpy.array(max(zip(p, f), key=lambda x:x[1])[0])*numpy.array(model.q)) for p, f in zip(pop, fit)]
y_mean = [numpy.mean([numpy.sum(numpy.array(i)*numpy.array(model.q)) for i in p]) for p in pop]
ax0.plot(g, len(g)*[max_Q], "r--", alpha=0.7, label="Melhor solução possível")
ax0.plot(g, y_max, color='red', alpha=0.7, label='Melhor indivíduo')
ax0.plot(g, y_mean, color='blue', alpha=0.7, label='Média da população')
ax0.legend(loc='lower right')
ax1.set_xlabel('Gerações')
ax1.set_ylabel('Custo total dos projetos selecionados (M)')
y_max  = [numpy.sum(numpy.array(max(zip(p, f), key=lambda x:x[1])[0])*numpy.array(model.c)) for p, f in zip(pop, fit)]
y_mean = [numpy.mean([numpy.sum(numpy.array(i)*numpy.array(model.c)) for i in p]) for p in pop]
ax1.plot(g, len(g)*[model.V], "r--", alpha=0.7, label="Verba disponível")
ax1.plot(g, y_max, color='red', alpha=0.7, label='Melhor indivíduo')
ax1.plot(g, y_mean, color='blue', alpha=0.7, label='Média da população')
ax1.legend(loc='upper right')

plt.tight_layout()
plt.show()