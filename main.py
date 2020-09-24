import functions as f
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from sklearn.datasets import make_circles

def calcular_topology(N,max,num):
    L = [N, 1]
    bandera = False
    for i in range(num):
        if L[i] == max:
            bandera = True
        if bandera:
            n_neuronas = int(L[i] / 2)
            L.insert(i + 1, n_neuronas)
        else:
            n_neuronas = int(L[i] * 2)
            L.insert(i + 1, n_neuronas)

    return L

#


#Creamos el DataSET
n = 500;
p = 2;

X,Y = make_circles(n_samples=n,factor=0.5,noise=0.05)
Y  = Y[:,np.newaxis]

print(X.shape)
print(Y.shape)



plt.scatter(X[Y[:,0] == 0,0],X[Y[:,0] == 0,1],c="skyblue")
plt.scatter(X[Y[:,0] == 1,0],X[Y[:,0] == 1,1],c="salmon")
plt.axis("equal")

plt.grid()
plt.show()

max = 16
for i in range(1,7):
    topology = calcular_topology(p,max,i)
    k = 0.01
    for i in range(10):
        f.red_neuronal(X,Y,topology,k)
        k+= 0.01