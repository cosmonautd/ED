import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple as nt

# Código modificado de https://www.peculiar-coding-endeavours.com/2018/mlp_vs_cnn/

# Definição do nível de log do Python e TensorFlow
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importação de modelos, camadas, datasets e utilidades do Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

# Importação de otimizações para a CNN
from keras.layers import BatchNormalization
from keras.optimizers import rmsprop
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

# Função para construção da CNN otimizada
def build_optimized_cnn():
    # Organização sequencial de camadas
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay), 
                     input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    # Compilação do modelo. Definição da função de perda e algoritmo de treinamento.
    optimized_rmsprop = rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer=optimized_rmsprop,
                  metrics=["accuracy"])
    return model

# Construção da CNN otimizada
optimized_cnn = build_optimized_cnn()

# Carregamento da melhor combinação de pesos salva
try:
    optimized_cnn_path_best = "saved_weights/optimized_cifar10_cnn_best.hdf5"
    optimized_cnn.load_weights(optimized_cnn_path_best)
except OSError:
    print("Arquivo de pesos treinados não encontrado")
    quit()

# Importação do OpenCV
import cv2

# Carregamento de uma imagem de test (classe 7)
# Redimensionamento para adequação à entrada da CNN
image = cv2.imread('test.jpg')
image = image.reshape((1,32,32,3))

# Obtenção da saída da rede
output = optimized_cnn.predict(image)
# Obtenção da classe
class_ = np.argmax(output)
print(class_)