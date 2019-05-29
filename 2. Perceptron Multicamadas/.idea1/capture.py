import warnings
warnings.filterwarnings(action='ignore')

import cv2
import keras
import numpy
import scipy.stats
import multiprocessing
import warnings
import collections
import shapelib

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

# Método 2 de extração de características de forma,
# baseado na Transformada de Fourier
method_2 = 'Fourier descriptors'

# Definição do método de extração de características de forma a ser usado
method = method_0

# Caso o método de descrição definido seja baseado em redes neurais randomizadas
if method in [method_0, method_1]:
    # Cálculo do tamanho do descritor
    descriptor_size = sum([n+1 for _,_,n in method])
    # Formação da pilha de descritores a ser utilizada, técnica
    # baseada no artigo https://www.sciencedirect.com/science/article/pii/S0925231218306842
    stack = [shapelib.ContourDescriptor(mode=m[0], params=(m[1],), neurons=m[2]) for m in method]
    descriptor = shapelib.StackedContourDescriptor(stack)
# Caso o método de descrição definido seja baseado na Transformada de Fourier
elif method == method_2:
    # Cálculo do tamanho do descritor
    descriptor_size = 8

# Exibição do tamanho do descritor
print('Descriptor size: %d' % descriptor_size)

# Carrega a arquitetura da rede neural de segmentação de pele
with open('model_s.json', 'r') as json_file:
    model_json = json_file.read()
    model_s = keras.models.model_from_json(model_json)
# Carrega os pesos da rede neural de segmentação de pele
model_s.load_weights("model_s.h5")

# Carrega a arquitetura da rede neural de reconhecimento de gestos
with open('model_g.json', 'r') as json_file:
    model_json = json_file.read()
    model_g = keras.models.model_from_json(model_json)
# Carrega os pesos da rede neural de reconhecimento de gestos
model_g.load_weights("model_g.h5")

# Rótulos das classes e suas correspondências com os neurônios da camada de 
# saída da rede neural de reconhecimento de gestos
# O valor -1 não é um neurônio. Significa apenas a ausência da classificação
class_ = {
    -1: ' ',
     0: '0',
     1: '1',
     2: '2',
     3: '3',
     4: 'hang loose'
}

# Lista para armazenamento dos últimos 10 gestos reconhecidos
g_ = collections.deque(maxlen=10)

# Kernels para processamento morfológico da forma, redução de ruído, etc
kernel_1 = numpy.ones((3,3), numpy.uint8)
kernel_2 = numpy.ones((5,5), numpy.uint8)

# Inicialização de captura de vídeo
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Loop principal
while(True):
    # Captura de um frame
    _, frame = video.read()

    # Desenha a região de interesse (um quadrado 300x300) + borda de 2px
    cv2.rectangle(frame, (50-2, 50-2), (350+2, 350+2), (0,255,0), 2)

    # Seleciona a região de interesse e salva uma cópia
    roi = frame[50:350,50:350]
    raw = roi.copy()

    # Redimensionamento da região de interesse para acelerar o processamento
    roi = cv2.resize(roi, (0,0), fx=0.5, fy=0.5)
    h, w, d = roi.shape
    # Transformação dos pixels da região de interesse em um vetor de pixels
    # para entrada na rede neural de segmentação de pele
    x = roi.reshape(h*w, d)

    # Predição do modelo de segmentação de pele
    y = model_s.predict(x)

    # A segmentação é feita com base na saída do primeiro neurônio,
    # que indica a probabilidade de determinado pixel representar pele
    segm = y[:,0]
    # Pixels que ativam o primeiro neurônio com intensidade menor 
    # que 0.7 são descartados
    idx = numpy.argwhere(segm < 0.7)
    segm[idx] = 0

    # Os valores restantes são escalonadas para o intervalo padrão
    # de representação de 8 bits de pixels em escala de cinza
    # e são redimensionados para sua representação de imagem
    segm = segm*255
    segm = segm.reshape(h, w, 1)
    
    # Processamento morfológico da segmentação para redução de ruído
    # através de erosão, fechamento e dilatação
    segm = cv2.erode(segm, kernel_1, iterations=1)
    segm = cv2.morphologyEx(segm, cv2.MORPH_CLOSE, kernel_2)
    segm = cv2.dilate(segm, kernel_1, iterations=2)

    # Variável para armazenamento da classe de gesto identificada no frame atual
    g = -1

    # Alteração do tipo de dado da segmentação para valores uint8
    segm = segm.astype(numpy.uint8)
    # Limiarização da segmentação para 0 (não pele) e 1 (pele)
    _, segm = cv2.threshold(segm, 125, 255, 0)
    # Extração de contornos
    contours, _ = cv2.findContours(segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Caso haja contornos
    if len(contours) > 0:
        # Extrai o maior contorno identificado
        main_contour = max(contours, key=lambda x:len(x))
        main_contour = numpy.reshape(main_contour, (len(main_contour), 2))
        # Caso o tamanho do contorno seja maior que um determinado limiar
        if len(main_contour) > 100:      
            # Extrai as características do contorno de acordo com o método especificado
            # Caso o método de descrição definido seja baseado em redes neurais randomizadas
            if method in [method_0, method_1]:
                features = descriptor.extract_contour_features(contour=main_contour)
            # Caso o método de descrição definido seja baseado na Transformada de Fourier
            elif method == method_2:
                main_contour = main_contour.reshape(main_contour.shape[0], 1, main_contour.shape[1])
                main_contour = numpy.asarray(main_contour, numpy.float32)
                features = cv2.ximgproc.fourierDescriptor(main_contour, 0, int(descriptor_size/2)).flatten()
            # Predição do gesto representado pelo contorno
            out = model_g.predict(numpy.array([features]))
            # Armazenamento do gesto classificado
            g_.append(numpy.argmax(out))
            # Cálculo do gesto mais provável de acordo com as últimas classificações
            g = scipy.stats.mode(g_)[0][0]
            # Alteração do número de canais da segmentação e desenho do contorno identificado
            segm = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(segm, [main_contour], 0, (0,255,0), 2)
        else:
            # Alteração do número de canais da segmentação
            segm = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
    else:
        # Alteração do número de canais da segmentação
        segm = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
            
    # Exibição da segmentação resultante
    # Redimensiona a região de interesse para seu tamanho original e a sobrepõe ao frame
    segm = cv2.resize(segm, (0,0), fx=2.0, fy=2.0)
    frame[50:350,50:350] = segm
    # Escreve o texto correspondente à classe identificada no canto superior esquerdo
    cv2.putText(frame, class_[g], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Exibe o frame
    cv2.imshow('F', frame)
    # Caso a tecla q seja pressionada, finaliza o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha as janelas abertas
video.release()
cv2.destroyAllWindows()