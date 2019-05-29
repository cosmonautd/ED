import os
import warnings

# Filtro para warnings irritantes
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import sys
import keras
import numpy
import matplotlib.pyplot as plt

def show_image(images):
    """ Exibe uma lista de imagens
    """
    n = len(images)
    if n == 1:
        fig, (ax0) = plt.subplots(ncols=1)
        ax0.imshow(images[0], cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_ticks([])
        ax0.axes.get_yaxis().set_visible(False)
    else:
        fig, axes = plt.subplots(ncols=n, figsize=(4*n, 4))
        for ax, image in zip(axes, images):
            ax.imshow(image, cmap='gray', interpolation='bicubic')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show()

if os.path.exists(sys.argv[1]):
    path = sys.argv[1]
else: quit()

# Carrega a arquitetura da rede neural de segmentação de pele
with open('model_s.json', 'r') as json_file:
    model_json = json_file.read()
    model_s = keras.models.model_from_json(model_json)
# Carrega os pesos da rede neural de segmentação de pele
model_s.load_weights('model_s.h5')

image = cv2.imread(path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, d = image.shape
# Transformação dos pixels da região de interesse em um vetor de pixels
# para entrada na rede neural de segmentação de pele
x = image.reshape(h*w, d)

# Predição do modelo de segmentação de pele
y = model_s.predict(x)

# A segmentação é feita com base na saída do primeiro neurônio,
# que indica a probabilidade de determinado pixel representar pele
segm = y[:,0]
# Pixels que ativam o primeiro neurônio com intensidade menor 
# que 0.8 são descartados
idx = numpy.argwhere(segm < 0.8)
segm[idx] = 0

# Os valores restantes são escalonadas para o intervalo padrão
# de representação de 8 bits de pixels em escala de cinza
# e são redimensionados para sua representação de imagem
segm = segm*255
segm = segm.reshape(h, w, 1)

# Alteração do número de canais da segmentação
segm = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)

show_image([image_rgb, segm])