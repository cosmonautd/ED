import numpy
import string
import matplotlib.pyplot as plt

# Configuração do gerador aleatório para reproducibilidade
numpy.random.seed(1)

W = 300
p = numpy.random.randint(1, 100, 25)
w = numpy.random.randint(1, 100, 25)
N = len(p)
 
# Função de inicialização da população
def init(pop_size):
    population = list()
    for _ in range(pop_size):
        individual = list(numpy.random.randint(0, 2, N))
        population.append(individual)
    return population

# Função de aptidão
def fit(population):
    fitness = list()
    for individual in population:
        f = 1
        fitness.append(f)
    return fitness

# Função de seleção
def selection(population, fitness, n):
    # Método da roleta
    def roulette():
        # Obtenção dos índices de cada indivíduo da população
        idx = numpy.arange(0, len(population))
        # Cálculo das probabilidades de seleção com base na aptidão dos indivíduos
        probabilities = fitness/numpy.sum(fitness)
        # Escolha dos índices dos pais
        parents_idx = numpy.random.choice(idx, size=n, p=probabilities)
        # Escolha dos pais com base nos índices selecionados
        parents = numpy.take(population, parents_idx, axis=0)
        # Organiza os pais em pares
        parents = [(list(parents[i]), list(parents[i+1])) for i in range(0, len(parents)-1, 2)]
        return parents
    # Método do torneio
    def tournament(K):
        # Lista dos pais
        parents = list()
        # Obtenção dos índices de cada indivíduo da população
        idx = numpy.arange(0, len(population))
        # Seleção de um determinado número n de pais
        for _ in range(n):
            # Seleciona K indivíduos para torneio
            turn_idx = numpy.random.choice(idx, size=K)
            # Seleciona o indivíduo partipante do torneio com maior aptidão
            turn_fitness = numpy.take(fitness, turn_idx, axis=0)
            argmax = numpy.argmax(turn_fitness)
            # Adiciona o indivíduo selecionado à lista de pais
            parents.append(population[argmax])
        # Organiza os pais em pares
        parents = [(list(parents[i]), list(parents[i+1])) for i in range(0, len(parents)-1, 2)]
        return parents
    # Método do Ranking
    def ranking():
        raise NotImplementedError
    # Escolha do método de seleção
    return roulette()

# Função de cruzamento
def crossover(parents, crossover_rate):
    # Lista de filhos
    children = list()
    # Iteração por todos os pares de pais
    for pair in parents:
        parent1 = pair[0]
        parent2 = pair[1]
        children.append(parent1)
        children.append(parent2)
    return children

# Função de mutação
def mutation(children, mutation_rate):
    # Mutação pode ocorrer em qualquer dos filhos
    for i, child in enumerate(children):
        pass
    return children

# Função de critério de parada
def stop():
    return False

# Função de elitismo
def elitism(population, fitness, n):
    # Seleciona n indivíduos mais aptos
    return [e[0] for e in sorted(zip(population, fitness),
                key=lambda x:x[1], reverse=True)[:n]]

# https://codereview.stackexchange.com/questions/20569/dynamic-programming-knapsack-solution
# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def maxknapsack():
    K = [[0 for x in range(W+1)] for x in range(N+1)]
    # Build table K[][] in bottom up manner
    for i in range(N+1):
        for w_ in range(W+1):
            if i==0 or w_==0:
                K[i][w_] = 0
            elif w[i-1] <= w_:
                K[i][w_] = max(p[i-1] + K[i-1][w_-w[i-1]],  K[i-1][w_])
            else:
                K[i][w_] = K[i-1][w_]
    return K[N][W]
