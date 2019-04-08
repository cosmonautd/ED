import warnings
warnings.filterwarnings(action='ignore')

import cv2
import numpy
import keras
import tensorflow
import matplotlib.pyplot as plt

# Definição fixa das sementes dos geradores aleatórios
# para facilitar a reprodução dos resultados
numpy.random.seed(1)
tensorflow.set_random_seed(1)

# Leitura da base de dados de segmentação de pele
# Cada linha contém uma amostra
# Os três primeiros números são os valores B, G e R de um pixel
# O quarto número corresponde à classe do pixel: 1 para pele, 2 para não-pele
# 50859 amostras de pele e 194198 de não-pele
samples = list()
with open('skin.txt') as skin:
    for row in skin.readlines():
        samples.append(list(map(int, row.split())))
# Representação do dataset como um array numpy
dataset = numpy.array(samples)
# Realiza uma permutação aleatória das amostras
numpy.random.shuffle(dataset)

# Número de amostras
n = len(dataset)
# Número de dimensões dos vetores de entrada
d = 3

# Preenchimento do vetor de amostras X e suas classes Y
# A subtração de 1 no valor de Y serve para normalizar os índices
# das classes para 0 e 1
X = dataset[:,:d]
Y = dataset[:,d:] - 1
# Alteração da codificação do vetor de classes para one-hot-encoding
# 0 torna-se [1, 0] e 1 torna-se [0, 1]
Y = keras.utils.to_categorical(Y)

# Instanciação de um modelo sequencial;
# Este modelo é uma pilha de camadas de neurônios;
# Sua construção é feita através da adição sequencial de camadas,
# primeiramente a camada de entrada, depois as camadas e ocultas e, 
# enfim, a camada de saída;
# Neste exemplo, a classe Dense representa camadas totalmente conectadas
model = keras.models.Sequential([
    keras.layers.Dense(5, activation='sigmoid', input_shape=(d,)),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(5, activation='sigmoid'),
    keras.layers.Dense(2, activation='softmax')
])

# Compilação do modelo
# Definição do algoritmo de otimização e da função de perda
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
# Executa o algoritmo de otimização, ajustando os pesos das conexões
# da rede neural com base nos valores de entrada X e saída Y, usando
# a função de perda como forma de verificar o quão corretas são suas
# predições durante o treinamento. Realiza 20 passagens pelo conjunto
# de treinamento. Utiliza 20% dos conjuntos X e Y como validação.
history = model.fit(X, Y, epochs=20, validation_split=0.2)

# Visualização da evolução da perda sobre os conjuntos de 
# treinamento e validação
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perda do modelo de segmentação de pele')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper right')

# Visualização da evolução da acurácia sobre os conjuntos de 
# treinamento e validação
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Acurácia do modelo de segmentação de pele')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='lower right')
plt.show()

# Salva a arquitetura da rede em um arquivo JSON
model_json = model.to_json()
with open('model_s.json', 'w') as json_file:
    json_file.write(model_json)

# Salva os pesos da rede em um arquivo HDF5
model.save_weights("model_s.h5")