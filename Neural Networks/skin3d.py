import os
import warnings

# Filtro para warnings irritantes
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy
import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definição fixa das sementes dos geradores aleatórios
# para facilitar a reprodução dos resultados
numpy.random.seed(1)

samples = list()

# Leitura da base de dados de segmentação de pele
# Cada linha contém uma amostra
# Os três primeiros números são os valores B, G e R de um pixel
# O quarto número corresponde à classe do pixel: 1 para pele, 2 para não-pele
# 50859 amostras de pele e 194198 de não-pele
# Link: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
with open('skin.txt') as skin:
    for row in skin.readlines():
        samples.append(list(map(int, row.split())))
        samples[-1][-1] += -1

# Carrega a arquitetura da rede neural de segmentação de pele
with open('model_s.json', 'r') as json_file:
    model_json = json_file.read()
    model_s = keras.models.model_from_json(model_json)
# Carrega os pesos da rede neural de segmentação de pele
model_s.load_weights("model_s.h5")

# Representação do dataset como um array numpy
dataset = numpy.array(samples)
# Realiza uma permutação aleatória das amostras
numpy.random.shuffle(dataset)
# Número de amostras e dimensões dos vetores de entrada
N, d = dataset.shape[0], dataset.shape[1] - 1
# Número de amostras a serem utilizadas no plot
n = int(0.05*N)
# Trucamento do dataset para 5% das amostras
dataset = dataset[:n,:]

# Plot nas dimensões originais das amostras
# Instanciação de uma figura 3D no matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Obtenção das dimensões RGB das amostras
B = dataset[:,0]
G = dataset[:,1]
R = dataset[:,2]
C = dataset[:,3]

# Scatter das amostras originais
ax.scatter(R, G, B, c=C, alpha=0.3)

# Plot nas dimensões finais da MLP
# Instanciação de uma figura no matplotlib
fig = plt.figure()

# Obtenção da saída da MLP
y = model_s.predict(dataset[:,:d])

# Obtenção das dimensões das saídas MLP
D1 = y[:,0]
D2 = y[:,1]

# Scatter das representações finais da MLP
plt.scatter(D1, D2, c=C, alpha=0.3)

# Exibição das figuras
plt.show()