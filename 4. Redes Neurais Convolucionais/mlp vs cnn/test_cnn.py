import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple as nt

# Código modificado de https://www.peculiar-coding-endeavours.com/2018/mlp_vs_cnn/

# Definição do nível de log do Python e TensorFlow
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Criação de diretório para armazenar os pesos treinados
if not os.path.exists('saved_weights'): os.mkdir('saved_weights')

# Definição dos dados de treinamento, validação e teste como uma tupla
Data = nt("Data", "x_train y_train x_valid y_valid x_test y_test")

# Função para visuzalizar exemplos do dataset
def visualize_data(data):
    images_to_show = 50
    per_row = 10
    fig = plt.figure(figsize=(20,5))
    i = 0
    for j in range(len(x_train)):
        pos = (i // per_row, ((i % per_row) + per_row) % per_row)
        if pos[1] == data.y_train[j]:
            ax = plt.subplot2grid((int(images_to_show / per_row), per_row),
                                  pos, xticks=[], yticks=[])
            ax.imshow(np.squeeze(data.x_train[j]))
            i += 1
        if i == images_to_show: break
    plt.show()

# Função para plot dos gráficos de acurácia e perda durante o treinamento
def visualize_training(hist):
    # Plot da acurácia
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show()
    # Plot da perda
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

# Importação de modelos, camadas, datasets e utilidades do Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

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
def build_cnn(data, categories):
    # Organização sequencial de camadas
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu",
                     input_shape=data.x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(categories, activation="softmax"))
    
    # Compilação do modelo. Definição da função de perda e algoritmo de treinamento.
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop",
                  metrics=["accuracy"])
    return model

# Carregamento dos dados de treinamento e teste.
# Download dos dados se for a primeira execução.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Truncamento do dataset. As primeiras 25000 imagens serão usadas no treinamento.
x_train, y_train = x_train[:25000], y_train[:25000]
# Impressão dos rótulos presentes e contagem dos elementos de cada rótulo.
print(np.unique(y_train, return_counts=True))
# Definição da tupla de dados
data = Data(x_train, y_train, None, None, x_test, y_test)

# Visualização de amostras do dataset. Cada coluna representa uma classe.
visualize_data(data)

# Pré-processamento dos dados e impressão das dimensões do dataset
categories = len(np.unique(data.y_train))
processed_data = preprocess(data, categories)
print("Shape of x_train pre-processing: ", data.x_train.shape)
print("Shape of y_train pre-processing: ", data.y_train.shape)
print("Shape of x_train post-processing: ", processed_data.x_train.shape)
print("Shape of y_train post-processing: ", processed_data.y_train.shape)
print("Shape of x_valid post-processing: ", processed_data.x_valid.shape)
print("Shape of y_valid post-processing: ", processed_data.y_valid.shape)
print("Shape of x_test post-processing: ", processed_data.x_test.shape)
print("Shape of y_test post-processing: ", processed_data.y_test.shape)

# Construção da CNN
cnn = build_cnn(processed_data, categories)

# Impressão da arquitetura da CNN
print("CNN architecture:")
cnn.summary()

# Definição do caminho para salvamento dos pesos
cnn_weights_path = "saved_weights/cifar10_cnn_best.hdf5"

# Definição de um callback para salvamento da melhor combinação de pesos
checkpointer_cnn = ModelCheckpoint(cnn_weights_path, verbose=1, save_best_only=True)

# Treinamento da CNN
hist_cnn = cnn.fit(processed_data.x_train, processed_data.y_train, batch_size=32, 
                   epochs=20, validation_data=(processed_data.x_valid, processed_data.y_valid),
                   callbacks=[checkpointer_cnn])

# Plot dos gráficos de acurácia e perda
visualize_training(hist_cnn)

# Carregamento da melhor combinação de pesos obtida durante o treinamento
cnn.load_weights(cnn_weights_path)

# Cálculo da acurácia sobre o conjunto de teste
score_cnn = cnn.evaluate(processed_data.x_test, processed_data.y_test, verbose=0)

# Impressão da acurácia
print("Accuracy cnn: {0:.2f}%".format(score_cnn[1] * 100))