import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output



class neural_layer():
  def __init__(self,n_comn,n_neur,act_f):
    self.act_f = act_f
    self.b = np.random.rand(1,n_neur)*2 -1
    self.W = np.random.rand(n_comn,n_neur)*2 -1

def create_nn(topology,act_f):
  nn = []
  for l,layer in enumerate(topology[:-1]):
    nn.append(neural_layer(topology[l],topology[l+1],act_f))
  return nn

def train(neural_net,X,Y,l2_cost,lr=0.5,train = True):
    out = [(None,X)]
    for l,layer in enumerate(neural_net):
        z = out[-1][1]@neural_net[l].W +  neural_net[l].b
        a = neural_net[l].act_f[0](z)
        out.append((z,a))

    if train:
        deltas = []
        for l in reversed(range(0,len(neural_net))):
            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(neural_net)-1:
                deltas.insert(0,l2_cost[1](a,Y)*neural_net[l].act_f[1](a))

            else:
                deltas.insert(0,deltas[0]@_W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

            neural_net[l].b = neural_net[l].b - np.mean(deltas[0],axis = 0,keepdims = True) *lr
            neural_net[l].W = neural_net[l].W - out[l][1].T@deltas[0]*lr

    return out[-1][1]

def calculate_limit(X,porcentaje,equal=False):
    valores = [X[:, 0],X[:, 1]]

    limites = []
    for i in valores:
        i = np.sort(i,axis=0)
        menor = i[0];
        mayor = i[-1]
        diferencia = mayor - menor

        limites.append([menor-diferencia*porcentaje,mayor+diferencia*porcentaje])

    if equal:
        diferenciaX = limites[0][1] - limites[0][0]
        diferenciaY = limites[1][1] - limites[1][0]

        if diferenciaX>diferenciaY:
            dif = diferenciaX-diferenciaY
            limites[1][1]+=dif/2
            limites[1][0]-=dif/2

        elif diferenciaY>diferenciaX:
            dif = diferenciaY - diferenciaX
            limites[0][1] += dif / 2
            limites[0][0] -= dif / 2

    return limites

def test_network(X,Y,pY,neural_n,l2_cost,loss,l_limits,topology,lr,ts = 50,end_test = False):
    limX,limY = l_limits
    _x0 = np.linspace(limX[0], limX[1], ts)
    _x1 = np.linspace(limY[0], limY[1], ts)

    _Y = np.zeros((ts, ts))

    for i0, x0 in enumerate(_x0):
        for i1, x1 in enumerate(_x1):
            _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]

    if end_test:
        plt.subplot(211)
        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")

        plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
        plt.xlim(limX)
        plt.ylim(limY)
        titulo = "["
        for i in topology:
            titulo += str(i) + ","
        titulo += "] lr = " + str(lr)
        plt.title(titulo)

        clear_output(wait=True)

        plt.subplot(212)
        plt.plot(range(len(loss)), loss)
        plt.show()

    else:
        loss.append(l2_cost[0](pY, Y))




def red_neuronal(X,Y,topology,k):
    sigm = (lambda x: 1 / (1 + np.e ** (-x)),
            lambda x: x * (1 - x))

    l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
               lambda Yp, Yr: (Yp - Yr))

    neural_n = create_nn(topology, sigm)
    loss = []

    l_limits = calculate_limit(X, 0.1,equal=False)


    for i in range(2500):
        pY = train(neural_n, X, Y, l2_cost, lr=k)
        if i%25==0:
            test_network(X, Y, pY, neural_n, l2_cost, loss, l_limits,50, False)

    test_network(X, Y, pY, neural_n, l2_cost, loss, l_limits,topology,k, 50, True)

    # Entrenamos la red




