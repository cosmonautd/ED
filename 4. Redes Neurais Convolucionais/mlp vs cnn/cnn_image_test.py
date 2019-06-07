import os
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple as nt

# Importação de modelos, camadas, datasets e utilidades do Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

# Definição dos dados de treinamento, validação e teste como uma tupla
Data = nt("Data", "x_train y_train x_valid y_valid x_test y_test")

# Função de pré-processamento dos dados
def preprocess(data, categories):
    # Normalização dos valores dos pixels para o intervalo [0, 1]
    x_train = data.x_train.astype("float32") / 255
    x_test = data.x_test.astype("float32") / 255
    # Representação one-hot-encoding para os rótulos
    y_train = to_categorical(data.y_train, categories)
    y_test = to_categorical(data.y_test, categories)
    return Data(x_train[5000:], y_train[5000:],
                x_train[:5000], y_train[:5000],
                x_test, y_test)

# Função para construção da CNN
def build_cnn():
    # Organização sequencial de camadas
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu",
                     input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation="softmax"))
    
    # Compilação do modelo. Definição da função de perda e algoritmo de treinamento.
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop",
                  metrics=["accuracy"])
    return model


# Construção da CNN
cnn = build_cnn()

# Carregamento da melhor combinação de pesos salva
cnn_weights_path = "saved_weights/cifar10_cnn_best.hdf5"
cnn.load_weights(cnn_weights_path)

# Importação do OpenCV
import cv2

# Carregamento de uma imagem de test (classe 7)
# Redimensionamento para adequação à entrada da CNN
image = cv2.imread('test.jpg')
image = image.reshape((1,32,32,3))

# Obtenção da saída da rede
output = cnn.predict(image)
# Obtenção da classe
class_ = np.argmax(output)
print(class_)