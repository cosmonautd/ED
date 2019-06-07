import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple as nt

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

# Importação de otimizações para a CNN
from keras.layers import BatchNormalization
from keras.optimizers import rmsprop
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

# Função de pré-processamento dos dados otimizada
def optimized_preprocess(data, categories):
    # Normalização dos valores dos pixels com Z-score
    mean = np.mean(data.x_train, axis=(0,1,2,3))
    std = np.std(data.x_train, axis=(0,1,2,3))
    x_train = ((data.x_train - mean) / (std + 1e-7)).astype("float32")
    x_test = ((data.x_test - mean) / (std + 1e-7)).astype("float32")
    # Representação one-hot-encoding para os rótulos
    y_train = to_categorical(data.y_train, categories)
    y_test = to_categorical(data.y_test, categories)    
    return Data(x_train[5000:], y_train[5000:],
                x_train[:5000], y_train[:5000],
                x_test, y_test)

# Função de atualização da taxa de aprendizado
def learningrate_schedule(epoch):
    # Taxa de aprendização padrão de 0.001
    # A partir da 51ª época, taxa descresce para 0.0007
    # A partir da 101ª época, taxa descresce para 0.0005
    # A partir da 136ª época, taxa descresce para 0.0003
    # A partir da 176ª época, taxa descresce para 0.0001
    rate = 0.001
    if epoch > 175:
        rate = 0.0001
    elif epoch > 135:
        rate = 0.0003
    elif epoch > 100:
        rate = 0.0005
    elif epoch > 50:
        rate = 0.0007    
    return rate

# Função para construção da CNN otimizada
def build_optimized_cnn(data, categories):
    # Organização sequencial de camadas
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay), 
                     input_shape=data.x_train.shape[1:]))
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
    model.add(Dense(categories, activation="softmax"))

    # Compilação do modelo. Definição da função de perda e algoritmo de treinamento.
    optimized_rmsprop = rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer=optimized_rmsprop,
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
optimized_processed_data = optimized_preprocess(data, categories)
print("Shape of x_train pre-processing: ", data.x_train.shape)
print("Shape of y_train pre-processing: ", data.y_train.shape)
print("Shape of x_train post-processing: ", optimized_processed_data.x_train.shape)
print("Shape of y_train post-processing: ", optimized_processed_data.y_train.shape)
print("Shape of x_valid post-processing: ", optimized_processed_data.x_valid.shape)
print("Shape of y_valid post-processing: ", optimized_processed_data.y_valid.shape)
print("Shape of x_test post-processing: ", optimized_processed_data.x_test.shape)
print("Shape of y_test post-processing: ", optimized_processed_data.y_test.shape)

# Construção da CNN otimizada
optimized_cnn = build_optimized_cnn(optimized_processed_data, categories)

# Impressão da arquitetura da CNN otimizada
print("Optimized CNN architecture:")
optimized_cnn.summary()

# Gerador para executar aumento de dados com rotações, deslocamentos e espelhamentos
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.15,
                             height_shift_range=0.15, horizontal_flip=True)
datagen.fit(optimized_processed_data.x_train)

# Definição do caminho para salvamento dos pesos
optimized_cnn_path_best = "saved_weights/optimized_cifar10_cnn_best.hdf5"

# Definição de um callback para salvamento da melhor combinação de pesos
checkpointer_optimized_cnn = ModelCheckpoint(optimized_cnn_path_best, verbose=1, 
                                             save_best_only=True)

# Treinamento da CNN otimizada
batch_size = 64
hist_optimized_cnn = optimized_cnn.fit_generator(
                        datagen.flow(optimized_processed_data.x_train, 
                        optimized_processed_data.y_train, batch_size=batch_size), 
                        steps_per_epoch=optimized_processed_data.x_train.shape[0] // batch_size, epochs=250,
                        verbose=1, validation_data= (optimized_processed_data.x_valid,
                        optimized_processed_data.y_valid), callbacks=[checkpointer_optimized_cnn,
                        LearningRateScheduler(learningrate_schedule), EarlyStopping(min_delta=0.001, 
                        patience=40)])

# Plot dos gráficos de acurácia e perda
visualize_training(hist_optimized_cnn)

# Carregamento da melhor combinação de pesos obtida durante o treinamento
optimized_cnn.load_weights(optimized_cnn_path_best)

# Cálculo da acurácia sobre o conjunto de teste
score_optimized_cnn = optimized_cnn.evaluate(optimized_processed_data.x_test, 
                                             optimized_processed_data.y_test, verbose=0)

# Impressão da acurácia
print("Accuracy optimized cnn: {0:.2f}%".format(score_optimized_cnn[1] * 100))