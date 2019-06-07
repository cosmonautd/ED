import numpy
import string
import matplotlib.pyplot as plt

# Configuração do gerador aleatória para reproducibilidade
numpy.random.seed(1)

# Definição do número de cidades
N = 25

# Geração de rótulos e coordenadas aleatórias para N cidades
CITY_LABELS = list(range(N))
CITY_COORD = numpy.random.randint(0, 200, (N, 2))
CITY_DICT = {label : coord for (label, coord) in zip(CITY_LABELS, CITY_COORD)}

# Função de inicialização da população
def init(pop_size):
    # Método aleatório
    def random_permutation():
        population = list()
        for _ in range(pop_size):
            # Cada indivíduo é uma permutação aleatória do conjunto de cidades
            individual = list(numpy.random.permutation(CITY_LABELS))
            population.append(individual)
        return population
    # Método heurístico
    def heuristic():
        raise NotImplementedError
    # Escolha do método de inicialização da população
    return random_permutation()

# Função de aptidão
def fit(population):
    # Calcula a aptidão de todos os indivíduos da população
    fitness = list()
    for individual in population:
        # Cálculo da distância total do percurso definido pelo indivíduo
        distance = 0
        for i, city in enumerate(individual):
            s = CITY_DICT[individual[i-1]]
            t = CITY_DICT[individual[i]]
            distance += numpy.linalg.norm(s-t)
        # Definição da aptidão como o inverso da distância
        fitness.append(1/distance)
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
        parents = [(parents[i], parents[i+1]) for i in range(0, len(parents)-1, 2)]
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
        parents = [(parents[i], parents[i+1]) for i in range(0, len(parents)-1, 2)]
        return parents
    # Método do Ranking
    def ranking():
        raise NotImplementedError
    # Escolha do método de seleção
    return roulette()

# Função de cruzamento
def crossover(parents, crossover_rate=0.9):
    # Método de cruzamento ordenado
    def ordered():
        # Lista de filhos
        children = list()
        # Iteração por todos os pares de pais
        for pair in parents:
            # Cruzamento ocorre com determinada probabilidade 
            if numpy.random.random() < crossover_rate:
                # Cada par de pais gera dois filhos
                for (parent1, parent2) in [(pair[0], pair[1]), (pair[1], pair[0])]:
                    # Definição do segmento de corte
                    points = numpy.random.randint(0, len(parent1), 2)
                    start = min(points)
                    end   = max(points)
                    # Obtenção do segmento central
                    segment1 = [x for x in parent1[start:end]]
                    # Obtenção do segmento da direita
                    segment2 = [x for x in parent2[end:] if x not in segment1]
                    # Obtenção do segmento da esquerda
                    segment3 = [x for x in parent2[:end] if x not in segment1]
                    # Construção da representação do filho
                    child = segment3 + segment1 + segment2
                    # Adição do novo filho à lista de filhos
                    children.append(child)
            else:
                # Caso o cruzamento não ocorra, os pais permanecem na próxima geração
                children.append(pair[0])
                children.append(pair[1])
        return children
    # Escolha do método de cruzamento
    return ordered()

# Função de mutação
def mutation(children, mutation_rate=0.05):
    # Método de inversão de bit
    def bitflip():
        raise NotImplementedError
    # Método de troca de dois elementos
    def swap():
        # Mutação pode ocorrer em qualquer dos filhos
        for i, child in enumerate(children):
            # Mutação ocorre com determinada probabilidade
            if numpy.random.random() < mutation_rate:
                # Seleciona duas coordenadas aleatoriamente 
                [a, b] = numpy.random.randint(0, len(child), 2)
                # Realiza permutação dos valores
                children[i][a], children[i][b] = children[i][b], children[i][a]
        return children
    # Método de inversão de segmento
    def inversion():
        raise NotImplementedError
    # Método de embaralhamento de segmento
    def scramble():
        raise NotImplementedError
    # Escolha do método de mutação
    return swap()

# Função de critério de parada
def stop():
    # Nenhum critério de parada específico definido
    return False

# Função de elitismo
def elitism(population, fitness, n):
    # Seleciona n indivíduos mais aptos
    return [e[0] for e in sorted(zip(population, fitness),
                key=lambda x:x[1], reverse=True)[:n]]