import numpy
import string
import matplotlib.pyplot as plt

# V: A verba liberada pelo governo
V = 300

# N: A quantidade de projetos submetidos
# c: Os custos de cada projeto (em milhões)
# q: A estimativa da quantidade de pessoas beneficiadas por cada projeto (em milhões)
N = numpy.random.randint(20, 30)
c = list(numpy.random.randint(10, 99, N))
q = list(numpy.random.randint(10, 99, N))
 
# Função de inicialização da população
def init(pop_size):
    # A população é uma lista de indivíduos
    population = list()
    for _ in range(pop_size):
        # Cada indivíduo é uma lista binária de comprimento N
        # A presença do valor 1 no índice i indice que o projeto i faz 
        # parte do conjunto de projetos selecionados.
        individual = list(numpy.random.randint(0, 2, N))
        population.append(individual)
    return population

# Função de aptidão
def fit(population):
    # Lista de aptidões
    fitness = list()
    # A aptidão é calculada para todos os indíduos da população
    for individual in population:
        # TODO: Implementar o cálculo da aptidão do indivíduo
        if numpy.sum(numpy.array(individual)*numpy.array(c)) > V: f = numpy.min(q)
        else: f = numpy.sum(numpy.array(individual)*numpy.array(q))
        # Fim do cálculo da aptidão do indivíduo
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
        # TODO: Cruzamento ocorre com determinada probabilidade
        if numpy.random.random() < crossover_rate:
            # TODO: Implementar o método de cruzamento
            # Os pais (parent1 e parent2) são listas binárias de comprimento N
            # Definição do segmento de corte
            point = numpy.random.randint(0, len(parent1))
            # Construção da representação dos filhos
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            # Adição dos novos filhos à lista de filhos
            children.append(child1)
            children.append(child2)
        else:
            # Caso o cruzamento não ocorra, os pais permanecem na próxima geração
            children.append(parent1)
            children.append(parent2)
    return children

# Função de mutação
def mutation(children, mutation_rate):
    # Mutação pode ocorrer em qualquer dos filhos
    for i, child in enumerate(children):
        # TODO: Mutação ocorre com determinada probabilidade
        if numpy.random.random() < mutation_rate:
            # TODO: Implementar o método de mutação
            # Um filho (child) é uma lista binária de comprimento N
            # Seleciona uma coordenada aleatoriamente 
            point = numpy.random.randint(0, len(child))
            # Realiza um flip no bit
            children[i][point] = 1 if child[point] == 0 else 1
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
# Returns the maximum value that can be put in a knapsack of capacity V
def maxknapsack():
    K = [[0 for x in range(V+1)] for x in range(N+1)]
    for i in range(N+1):
        for w in range(V+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif c[i-1] <= w:
                K[i][w] = max(q[i-1] + K[i-1][w-c[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
    return K[N][V]