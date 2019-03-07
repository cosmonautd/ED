import numpy
import skfuzzy
import matplotlib.pyplot as plt
import seaborn as sns

t = 0
t_ = 0.001
p = 0
v = 0
a = 0
m = 1
g = -9.8
f = 0
e = 0
P = list()
T = list()
Q = list()

def fuzzy_controller(p_ref, p, v, f_prev):
    e = p - p_ref

    # Fuzzificação das variáveis de entrada
    e_N = skfuzzy.sigmf(numpy.array([e]), -3, -1)[0]
    e_Z = skfuzzy.trimf(numpy.array([e]), [-3,  0, 3])[0]
    e_P = skfuzzy.sigmf(numpy.array([e]), 3, 1)[0]

    v_N = skfuzzy.sigmf(numpy.array([v]), -3, -1)[0]
    v_Z = skfuzzy.trimf(numpy.array([v]), [-3,  0,  3])[0]
    v_P = skfuzzy.sigmf(numpy.array([v]), 3, 1)[0]

    # Aplicação das operações fuzzy codificadas nas regras do modelo
    R1 = min(e_N, v_N)
    R2 = min(e_Z, v_N)
    R3 = min(e_P, v_N)
    R4 = min(e_N, v_Z)
    R5 = min(e_Z, v_Z)
    R6 = min(e_P, v_Z)
    R7 = min(e_N, v_P)
    R8 = min(e_Z, v_P)
    R9 = min(e_P, v_P)

    # Processo de combinação das regras e implicação sobre os conjuntos fuzzy de saída
    IT = numpy.sqrt(R1**2 + R2**2 + R3**2 + R4**2)
    NC = numpy.sqrt(R5**2)
    DT = numpy.sqrt(R6**2 + R7**2 + R8**2 + R9**2)
    
    # Deffuzificação usando o método da média ponderada
    output = (-0.5*DT + 0*NC + 0.5*IT)/(DT + NC + IT)
    
    if f_prev == 0: f = output
    else: 
        f = f_prev + output*f_prev
        f = min(f, 1.5*abs(m*g))
        f = max(f, 0)

    return f, e

seconds = 10
for i in range(seconds*10**3):
    f, e = fuzzy_controller(4, p, v, f)
    F = m*g + f
    a = F/m
    v = v + a*t_
    p = max(p + v*t_, 0)
    if p == 0: v = 0
    P.append(p)
    T.append(t)
    Q.append(f)
    t += t_

plt.plot(T, len(T)*[4], "b--", alpha=0.8, label="Reference (m)")
plt.plot(T, P, "b", alpha=0.8, label="Position (m)")
plt.plot(T, Q, "r", alpha=0.8, label="Applied Force (N)")
plt.legend()
plt.show()
