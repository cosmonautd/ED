import warnings
warnings.filterwarnings(action='ignore')

import os
import cv2
import numpy
import keras
import tensorflow
import shapelib

numpy.random.seed(1)
tensorflow.set_random_seed(1)

method = [('neighborhood', 6, 6),
          ('neighborhood', 8, 2),
          ('neighborhood', 10, 2),
          ('neighborhood', 10, 3),
          ('contour_portion', 5, 6),
          ('contour_portion', 20, 2),
          ('angle', 5, 4),
          ('angle', 10, 2),
          ('angle', 15, 7),
          ('angle', 20, 5),
          ('angle_plus', 5, 6),
          ('angle_plus', 25, 7)]

print('Descriptor size: %d' % (sum([n+1 for _,_,n in method])))

stack = [shapelib.ContourDescriptor(mode=m[0], params=(m[1],), neurons=m[2]) for m in method]
descriptor = shapelib.StackedContourDescriptor(stack)

# Leitura da base de dados de segmentação de pele
samples = list()
for f in sorted([f for f in os.listdir('gestures') if f.endswith('.jpg')]):
    image = cv2.imread(os.path.join('gestures', f), 0)
    features = descriptor.extract_contour_features(image=image)
    samples.append(list(features) + [f.split('_')[0]])
dataset = numpy.array(samples)
numpy.random.shuffle(dataset)
# Número de amostras
n = len(dataset)
# Número de dimensões dos vetores de entrada
d = 64
# Preenchimento do vetor de amostras X e suas classes Y
X = dataset[:,:d]
Y = dataset[:,d:]
# Alteração da codificação do vetor de classes para one-hot-encoding
Y = keras.utils.to_categorical(Y)
# Instanciação de um modelo sequencial
model = keras.models.Sequential()

# Adição de camadas ao modelo
model.add(keras.layers.Dense(512, activation='sigmoid', input_shape=(d,)))
model.add(keras.layers.Dense(256, activation='sigmoid'))
model.add(keras.layers.Dense(128, activation='sigmoid'))
model.add(keras.layers.Dense(6, activation='softmax'))

# Compilação do modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(X, Y, epochs=500, validation_split=0.2)

# serialize model to JSON
model_json = model.to_json()
with open('model_g.json', 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_g.h5")