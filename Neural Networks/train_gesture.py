import warnings
warnings.filterwarnings(action='ignore')

import os
import cv2
import numpy
import keras
import tensorflow
import shapelib

# Definição fixa das sementes dos geradores aleatórios
# para facilitar a reprodução dos resultados
numpy.random.seed(1)
tensorflow.set_random_seed(1)

# Método 0 de extração de características de forma, 
# baseado no artigo https://www.sciencedirect.com/science/article/pii/S0925231218306842
method_0 = [('neighborhood', 2, 4),
            ('neighborhood', 4, 4),
            ('neighborhood', 6, 4),
            ('neighborhood', 8, 4),
            ('contour_portion',  5,  4),
            ('contour_portion', 10, 4),
            ('contour_portion', 15, 4),
            ('contour_portion', 20, 4)
]

# Método 1 de extração de características de forma, 
# baseado em novas pesquisas
method_1 = [('neighborhood', 6, 6),
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
            ('angle_plus', 25, 7)
]

# Definição do método de extração de características de forma a ser usado
method = method_0
# Cálculo e exibição do tamanho do descritor
descriptor_size = (sum([n+1 for _,_,n in method]))
print('Descriptor size: %d' % descriptor_size)

# Formação da pilha de descritores a ser utilizada, técnica
# baseada no artigo https://www.sciencedirect.com/science/article/pii/S0925231218306842
stack = [shapelib.ContourDescriptor(mode=m[0], params=(m[1],), neurons=m[2]) for m in method]
descriptor = shapelib.StackedContourDescriptor(stack)

# Lista para armazenar as amostras de gestos
samples = list()
# Tradução do significado original dos gestos para representação numérica
class_ = {
    't': 0,
    'd': 1,
    'v': 2,
    'w': 3,
    'y': 4
}

# Leitura da base de dados de gestos
for f in sorted([f for f in os.listdir('gestures') if f.endswith('.png')]):
    # Leitura da imagem
    image = cv2.imread(os.path.join('gestures', f), 0)
    # Extração das características de forma
    features = descriptor.extract_contour_features(image=image)
    # Armazenamento das características e suas classes na lista de amostras
    samples.append(list(features) + [class_[f.split('_')[0]]])

# Conversão das amostras para arrays numpy e embaralhamento
dataset = numpy.array(samples)
numpy.random.shuffle(dataset)

# Número de amostras
n = len(dataset)
# Número de dimensões dos vetores de entrada
d = descriptor_size

# Preenchimento do vetor de amostras X e suas classes Y
X = dataset[:,:d]
Y = dataset[:,d:]
# Alteração da codificação do vetor de classes para one-hot-encoding
Y = keras.utils.to_categorical(Y)

# Instanciação de um modelo sequencial;
# Este modelo é uma pilha de camadas de neurônios;
# Sua construção é feita através da adição sequencial de camadas,
# primeiramente a camada de entrada, depois as camadas e ocultas e, 
# enfim, a camada de saída;
# Neste exemplo, a classe Dense representa camadas totalmente conectadas
model = keras.models.Sequential([
    keras.layers.Dense(96, activation='sigmoid', input_shape=(d,)),
    keras.layers.Dense(5, activation='softmax')
])

# Compilação do modelo
# Definição do algoritmo de otimização e da função de perda
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
# Executa o algoritmo de otimização, ajustando os pesos das conexões
# da rede neural com base nos valores de entrada X e saída Y, usando
# a função de perda como forma de verificar o quão corretas são suas
# predições durante o treinamento. Realiza 10 passagens pelo conjunto
# de treinamento. Utiliza 20% dos conjuntos X e Y como validação.
model.fit(X, Y, epochs=50, validation_split=0.2)

# Salva a arquitetura da rede em um arquivo JSON
model_json = model.to_json()
with open('model_g.json', 'w') as json_file:
    json_file.write(model_json)

# Salva os pesos da rede em um arquivo HDF5
model.save_weights("model_g.h5")