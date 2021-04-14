import numpy as np
import pandas as pd
import utiles as Utility

act_f = (lambda x: 1/(1+np.exp(-x)) ,
            lambda x: x* (1-x))

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr)**2),
                lambda Yp, Yr: (Yp - Yr))

def test_snn(a0, w1, w2):
    z = np.dot(w1.T, a0)
    a1 = act_f[0](z)
    a2 = np.dot(w2.T, a1)
    return a2


def main():
    inpx = "test_x.csv"
    inpy = "test_y.csv"
    pesos = "pesos.npz"
    # Carga Configuracion
    DT, hn, mu, it = Utility.load_config()
    # Carga Data
    xv = Utility.csv_to_numpy(inpx)
    yv = Utility.csv_to_numpy(inpy)
    # Carga Pesos
    w1, w2 = Utility.load_w_npy(pesos)
    # Calculo de la red
    y_out = test_snn(xv,  w1, w2)
    # Metricas de Test
    metricas = Utility.metrics(y_out, yv, "test_metrica.csv")
    np.savetxt("antes.csv",np.c_[y_out],delimiter=",")
    yv = yv[:,np.newaxis]
    np.savetxt("test_costo.csv", np.c_[yv, y_out.T], delimiter=",", fmt="%.6f")


if __name__ == "__main__":
    main()
