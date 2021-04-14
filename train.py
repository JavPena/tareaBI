# Training SNN based on Pseudo-inverse

import pandas as pd
import numpy as np
import math
import random
import utiles as Utility
import math

# Calculate Pseudo-inverse


class neural_layer():

    def __init__(self, n_conn, n_neur):

        self.W = np.random.rand(n_conn, n_neur) * 2 - 1


def p_inversa(a1, ye, hn, C):
    ya = np.dot(ye, a1.T)
    ai = np.dot(a1, a1.T) + np.eye(int(hn))/C
    p_inv = np.linalg.pinv(ai)
    w2 = np.dot(ya, p_inv)
    return(w2)


def create_nn(topology):

    nn = []

    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1]))

    return nn


# Training SNN via Pseudo-inverse
def train(xe, ye,nh,mu,MaxIter):
    act_f = (lambda x: 1/(1+np.exp(-x)) ,
            lambda x: x* (1-x))
    l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr)**2),
                lambda Yp, Yr: (Yp - Yr))

    topology = [xe.shape[0],]
    Y = ye[:,np.newaxis]

    topology.append(nh)

    topology.append(1)

    mse=[]
    neural_network = create_nn(topology)
    for k in range(MaxIter):
        out = [(None, np.transpose(xe))]
        
        for l , layer in enumerate(neural_network):


        
            z = out[-1][1] @ neural_network[l].W 
            a = act_f[0](z)

            out.append((z,a))

        

        deltas = []

        
        
        for l in  reversed(range(0, len(neural_network))):

            z = out[l+1][0]
            a = out[l+1][1]
            
            if(l == len(neural_network) - 1):

                deltas.append( l2_cost[1](a,Y) * act_f[1](a))
            else:
                deltas.insert(0,deltas[0] @ _w.T * act_f[1](a))
            
            _w = neural_network[l].W

            neural_network[l].W = neural_network[l].W - out[l][1].T @ deltas[0] * mu

            mse = l2_cost[1](out[-1][1],Y)


    
    return(neural_network[0].W, neural_network[1].W,mse)


def main():
    inpx = "train_x.csv"
    inpy = "train_y.csv"
    # Carga Configuracion
    DT, hn, mu, it = Utility.load_config()
    # Carga Data
    xe = Utility.csv_to_numpy(inpx)
    ye = Utility.csv_to_numpy(inpy)
    # Entrena
    w1, w2, mse = train(xe, ye, hn, mu , it)

    # Guarda Pesos
    Utility.save_w_npy(w1, w2)




if __name__ == '__main__':
    main()
