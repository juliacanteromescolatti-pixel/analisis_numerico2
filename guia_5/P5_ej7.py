import numpy as np
import matplotlib.pyplot as plt

def filtrar_ruido_svd(t, A, k=None):
    m, n = A.shape
    p = min(m, n)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Caida abrupta: mayor cociente S[i]/S[i+1]; se queda con k = i+1 valores
    if k is None:
        cocientes = S[:p-1] / S[1:p]
        k = int(np.argmax(cocientes)) + 1

    # A_k = sum_{j<k} S[j] u_j v_j^T  (igual que en im_aprox_svd)
    A_k = np.zeros((m, n))
    for j in range(k):
        A_k = A_k + S[j]*np.outer(U[:, j], Vt[j, :])

    # Valores singulares
    plt.figure()
    plt.semilogy(range(1, p+1), S, "o-")
    plt.axvline(k + 0.5, color="r", linestyle="--", label=f"k = {k}")
    plt.xlabel("i")
    plt.ylabel("sigma_i")
    plt.title("Valores singulares")
    plt.legend()
    plt.grid(True)

    # Senal ruidosa vs senal limpia, un grafico por sensor
    fig, ejes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    for s in range(n):
        ax = ejes[s // 5, s % 5]
        ax.plot(t, A[:, s], "r.-", alpha=0.5, label="ruidosa")
        ax.plot(t, A_k[:, s], "b-", label="filtrada")
        ax.set_title(f"Sensor {s+1}")
        ax.grid(True)
    ejes[0, 0].legend()
    fig.supxlabel("t")
    plt.tight_layout()
    plt.show()

    err = np.linalg.norm(A - A_k, 2)   # = S[k]
    print("k =", k)
    print("||A - A_k||_2 =", err)
    return A_k, k
ruta = "/Kmom/Home/analisis_numerico2/guia_5/datos_sensores.txt"

datos = np.loadtxt(ruta)

t = datos[:, 0]     # tiempo, R^30
A = datos[:, 1:]    # mediciones, R^{30x10}

A_k, k = filtrar_ruido_svd(t, A)