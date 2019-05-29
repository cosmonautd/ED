import numpy
import skfuzzy
import matplotlib.pyplot as plt
import seaborn as sns

# Variável de tempo (s), intervalo de tempo (s) e constante de segundos
t = 0
t_ = 0.001
seconds = int(1/t_)

# Referência da posição (m)
r = 5

# Posição (m), velocidade (m/s), aceleração (m/s²) e massa (kg) do sistema
p = 0
v = 0
a = 0
m = 1

# Aceleração da gravidade (m/s²)
g = -9.8

# Lista para armazenar as posições (P) e forças aplicadas pelo controlador (F)
# a cada instante de tempo (T)
P = list()
F = list()
T = list()

class FuzzyController:

    def __init__(self, ref, view=False):
        # Referência, força aplicada e força máxima capaz de ser aplicada
        # pelo controlador
        self.ref = ref
        self.f = 0
        self.f_max = 1.5*abs(m*g)
        # Intervalos das variáveis de entrada
        self.position_error_range = numpy.arange(-100, 100, 0.01)
        self.velocity_range = numpy.arange(-100, 100, 0.01)
        self.accel_range = numpy.arange(-100, 100, 0.01)
        # Intervalo da variável de saída
        self.output_range = numpy.arange(-1, 1, 0.01)
        # Definição das funções de pertinência da variável de erro de posição
        self.e_N = skfuzzy.sigmf(self.position_error_range, -5, -1)
        self.e_Z = skfuzzy.gaussmf(self.position_error_range, 0, 2)
        self.e_P = skfuzzy.sigmf(self.position_error_range, 5, 1)
        # Definição das funções de pertinência da variável de velocidade
        self.v_N = skfuzzy.sigmf(self.velocity_range, -5, -1)
        self.v_Z = skfuzzy.gaussmf(self.velocity_range, 0, 2)
        self.v_P = skfuzzy.sigmf(self.velocity_range, 5, 1)
        # Definição das funções de pertinência da variável de aceleração
        self.a_N = skfuzzy.sigmf(self.accel_range, -4, -1)
        self.a_Z = skfuzzy.gaussmf(self.accel_range, 0, 2)
        self.a_P = skfuzzy.sigmf(self.accel_range, 4, 1)
        # Definição das funções de pertinência da variável de saída
        self.o_N = skfuzzy.sigmf(self.output_range, -0.5, -10)
        self.o_Z = skfuzzy.gaussmf(self.output_range, 0, 0.2)
        self.o_P = skfuzzy.sigmf(self.output_range, 0.5, 10)

        # Visualização das funções de pertinência das entradas e saídas
        if view:
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

            ax0.plot(self.position_error_range, self.e_N, 'b', linewidth=1.5, label='Negativo')
            ax0.plot(self.position_error_range, self.e_Z, 'g', linewidth=1.5, label='Zero')
            ax0.plot(self.position_error_range, self.e_P, 'r', linewidth=1.5, label='Positivo')
            ax0.set_title('Erro de posição')
            ax0.legend()

            ax1.plot(self.velocity_range, self.v_N, 'b', linewidth=1.5, label='Negativo')
            ax1.plot(self.velocity_range, self.v_Z, 'g', linewidth=1.5, label='Zero')
            ax1.plot(self.velocity_range, self.v_P, 'r', linewidth=1.5, label='Positivo')
            ax1.set_title('Velocidade')
            ax1.legend()

            ax2.plot(self.accel_range, self.a_N, 'b', linewidth=1.5, label='Negativo')
            ax2.plot(self.accel_range, self.a_Z, 'g', linewidth=1.5, label='Zero')
            ax2.plot(self.accel_range, self.a_P, 'r', linewidth=1.5, label='Positivo')
            ax2.set_title('Aceleração')
            ax2.legend()

            ax3.plot(self.output_range, self.o_N, 'b', linewidth=1.5, label='Redução')
            ax3.plot(self.output_range, self.o_Z, 'g', linewidth=1.5, label='Manutenção')
            ax3.plot(self.output_range, self.o_P, 'r', linewidth=1.5, label='Aumento')
            ax3.set_title('Saída')
            ax3.legend()

            for ax in (ax0, ax1, ax2, ax3):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            plt.tight_layout()
    
    def infer(self, p, v, a, view=False):
        # Cálculo do erro de posição
        e = p - self.ref - 0.05 # Correção de viés de 5,0 cm
        # Fuzzificação do erro de posição
        e_N = skfuzzy.interp_membership(self.position_error_range, self.e_N, e)
        e_Z = skfuzzy.interp_membership(self.position_error_range, self.e_Z, e)
        e_P = skfuzzy.interp_membership(self.position_error_range, self.e_P, e)
        # Fuzzificação da velocidade
        v_N = skfuzzy.interp_membership(self.velocity_range, self.v_N, v)
        v_Z = skfuzzy.interp_membership(self.velocity_range, self.v_Z, v)
        v_P = skfuzzy.interp_membership(self.velocity_range, self.v_P, v)
        # Fuzzificação da aceleração
        a_N = skfuzzy.interp_membership(self.accel_range, self.a_N, a)
        a_Z = skfuzzy.interp_membership(self.accel_range, self.a_Z, a)
        a_P = skfuzzy.interp_membership(self.accel_range, self.a_P, a)
        # Aplicação das operações fuzzy codificadas nas regras do modelo
        # Se o sistema está abaixo da referência, movendo-se para baixo e acelerando para baixo
        R01 = numpy.fmin(numpy.fmin(e_N, v_N), a_N)
        # Se o sistema está abaixo da referência, movendo-se para baixo e com aceleração zero
        R02 = numpy.fmin(numpy.fmin(e_N, v_N), a_Z)
        # Se o sistema está abaixo da referência, movendo-se para baixo e acelerando para cima
        R03 = numpy.fmin(numpy.fmin(e_N, v_N), a_P)
        # Se o sistema está abaixo da referência, sem movimento e acelerando para baixo
        R04 = numpy.fmin(numpy.fmin(e_N, v_Z), a_N)
        # Se o sistema está abaixo da referência, sem movimento e com aceleração zero
        R05 = numpy.fmin(numpy.fmin(e_N, v_Z), a_Z)
        # Se o sistema está abaixo da referência, sem movimento e acelerando para cima
        R06 = numpy.fmin(numpy.fmin(e_N, v_Z), a_P)
        # Se o sistema está abaixo da referência, movendo-se para cima e acelerando para baixo
        R07 = numpy.fmin(numpy.fmin(e_N, v_P), a_N)
        # Se o sistema está abaixo da referência, movendo-se para cima e com aceleração zero
        R08 = numpy.fmin(numpy.fmin(e_N, v_P), a_Z)
        # Se o sistema está abaixo da referência, movendo-se para cima e acelerando para cima
        R09 = numpy.fmin(numpy.fmin(e_N, v_P), a_P)
        # Se o sistema está na referência, movendo-se para baixo e acelerando para baixo
        R10 = numpy.fmin(numpy.fmin(e_Z, v_N), a_N)
        # Se o sistema está na referência, movendo-se para baixo e com aceleração zero
        R11 = numpy.fmin(numpy.fmin(e_Z, v_N), a_Z)
        # Se o sistema está na referência, movendo-se para baixo e acelerando para cima
        R12 = numpy.fmin(numpy.fmin(e_Z, v_N), a_P)
        # Se o sistema está na referência, sem movimento e acelerando para baixo
        R13 = numpy.fmin(numpy.fmin(e_Z, v_Z), a_N)
        # Se o sistema está na referência, sem movimento e com aceleração zero
        R14 = numpy.fmin(numpy.fmin(e_Z, v_Z), a_Z)
        # Se o sistema está na referência, sem movimento e acelerando para cima
        R15 = numpy.fmin(numpy.fmin(e_Z, v_Z), a_P)
        # Se o sistema está na referência, movendo-se para cima e acelerando para baixo
        R16 = numpy.fmin(numpy.fmin(e_Z, v_P), a_N)
        # Se o sistema está na referência, movendo-se para cima e com aceleração zero
        R17 = numpy.fmin(numpy.fmin(e_Z, v_P), a_Z)
        # Se o sistema está na referência, movendo-se para cima e acelerando para cima
        R18 = numpy.fmin(numpy.fmin(e_Z, v_P), a_P)
        # Se o sistema está acima da referência, movendo-se para baixo e acelerando para baixo
        R19 = numpy.fmin(numpy.fmin(e_P, v_N), a_N)
        # Se o sistema está acima da referência, movendo-se para baixo e com aceleração zero
        R20 = numpy.fmin(numpy.fmin(e_P, v_N), a_Z)
        # Se o sistema está acima da referência, movendo-se para baixo e acelerando para cima
        R21 = numpy.fmin(numpy.fmin(e_P, v_N), a_P)
        # Se o sistema está acima da referência, sem movimento e acelerando para baixo
        R22 = numpy.fmin(numpy.fmin(e_P, v_Z), a_N)
        # Se o sistema está acima da referência, sem movimento e com aceleração zero
        R23 = numpy.fmin(numpy.fmin(e_P, v_Z), a_Z)
        # Se o sistema está acima da referência, sem movimento e acelerando para cima
        R24 = numpy.fmin(numpy.fmin(e_P, v_Z), a_P)
        # Se o sistema está acima da referência, movendo-se para cima e acelerando para baixo
        R25 = numpy.fmin(numpy.fmin(e_P, v_P), a_N)
        # Se o sistema está acima da referência, movendo-se para cima e com aceleração zero
        R26 = numpy.fmin(numpy.fmin(e_P, v_P), a_Z)
        # Se o sistema está acima da referência, movendo-se para cima e acelerando para cima
        R27 = numpy.fmin(numpy.fmin(e_P, v_P), a_P)
        # Combinação das regras
        R = R09 + R15 + R17 + R18 + R23 + R24 + R26 + R27 # Redução
        M = R03 + R06 + R07 + R08 + R12 + R14 + R16 + R20 + R21 + R22 + R25 # Manutenção
        A = R01 + R02 + R04 + R05 + R10 + R11 + R13 + R19 # Aumento
        # Corte das funções de pertinência da variável de saída
        R = numpy.fmin(R, self.o_N)
        M = numpy.fmin(M, self.o_Z)
        A = numpy.fmin(A, self.o_P)
        # Agregação dos conjuntos fuzzy de saída
        aggregated = numpy.fmax(R, numpy.fmax(M, A))
        # Deffuzificação
        output = skfuzzy.defuzz(self.output_range, aggregated, 'centroid')
        # Atualização da força aplicada pelo sistema de controle
        if self.f == 0:
            self.f = output*self.f_max
        else:
            # Incrementa ou decrementa a força atual aplicada de acordo com a
            # saída do sistema fuzzy
            self.f += output*self.f
        # Impede que o sistema de controle aplique forças negativas
        self.f = max(self.f, 0)
        # Impede que o sistema de controle aplique uma força acima da máxima
        self.f = min(self.f, self.f_max)

        if view:
            # Visualização das funções de pertinência associadas aos conjuntos fuzzy de saída
            out_ = numpy.zeros_like(self.output_range)
            fig, ax0 = plt.subplots(figsize=(8, 3))

            ax0.fill_between(self.output_range, out_, R, facecolor='b', alpha=0.7)
            ax0.plot(self.output_range, self.o_N, 'b', linewidth=0.5, linestyle='--', )
            ax0.fill_between(self.output_range, out_, M, facecolor='g', alpha=0.7)
            ax0.plot(self.output_range, self.o_Z, 'g', linewidth=0.5, linestyle='--')
            ax0.fill_between(self.output_range, out_, A, facecolor='r', alpha=0.7)
            ax0.plot(self.output_range, self.o_P, 'r', linewidth=0.5, linestyle='--')
            ax0.set_title('Consequências das regras sobre o conjunto fuzzy de saída')

            for ax in (ax0,):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            plt.tight_layout()

            # Visualização da agregação dos conjuntos fuzzy de saída e da deffuzificação
            activation = skfuzzy.interp_membership(self.output_range, aggregated, output)
            fig, ax0 = plt.subplots(figsize=(8, 3))

            ax0.plot(self.output_range, self.o_N, 'b', linewidth=0.5, linestyle='--', )
            ax0.plot(self.output_range, self.o_Z, 'g', linewidth=0.5, linestyle='--')
            ax0.plot(self.output_range, self.o_P, 'r', linewidth=0.5, linestyle='--')
            ax0.fill_between(self.output_range, out_, aggregated, facecolor='Orange', alpha=0.7)
            ax0.plot([output, output], [0, output], 'k', linewidth=1.5, alpha=0.9)
            ax0.set_title('Funções de pertinência agregadas e resultado da deffuzificação')

            for ax in (ax0,):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            plt.tight_layout()

        return self.f

# Instanciação de um objeto FuzzyController
fc = FuzzyController(ref=r, view=True)
# Teste de inferência
fc.infer(r,-5, 0, view=True)

# Início da simulação com duração de 15 segundos
for i in range(15*seconds):
    # Realiza inferência fuzzy com base na posição, velocidade e aceleração atuais
    f = fc.infer(p,v,a)
    # Calcula a força vertical a ser aplicada sobre o veículo aéreo
    # Neste caso, a força da gravidade e a força aplicada pelo controlador
    f_ = m*g + f
    # Calcula a aceleração vertical resultante da aplicação da força
    a = f_/m
    # Atualiza a velocidade vertical do veículo aéreo
    v = v + a*t_
    # Atualiza a posição vertical do veículo aéreo
    p = max(p + v*t_, 0)
    # Caso o objeto atinja o chão, a velocidade cai para zero
    if p == 0: v = 0
    # Salva os valores de posição, tempo e força aplicada a cada instante
    P.append(p)
    T.append(t)
    F.append(f)
    # Atualiza a variável de tempo
    t += t_

# Plot do gráfico final com posição e força aplicada pelo controlador
# em função do tempo
plt.figure()
plt.plot(T, len(T)*[r], "b--", alpha=0.8, label="Referência (m)")
plt.plot(T, P, "b", alpha=0.8, label="Posição (m)")
plt.plot(T, F, "r", alpha=0.8, label="Força aplicada (N)")
plt.legend()

plt.show()