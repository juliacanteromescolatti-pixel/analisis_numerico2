import numpy as np
import matplotlib.pyplot as plt

def iris_svd(datos):
    X = datos[:, :-1]               # 4 atributos
    clases = datos[:, -1].astype(int)  # 0-setosa, 1-versicolour, 2-virginica

    # SVD reducida: U es 150x4
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    nombres = ["setosa", "versicolour", "virginica"]
    colores = ["tab:blue", "tab:orange", "tab:green"]

    for c in range(3):
        idx = (clases == c)
        plt.scatter(U[idx, 0], U[idx, 1], c=colores[c], label=nombres[c], s=20)

    plt.xlabel("U[:,0]")
    plt.ylabel("U[:,1]")
    plt.title("Iris: primeras dos columnas de U")
    plt.legend()
    plt.grid(True)
    plt.show()

    return U, S, Vt

# Si las clases vienen como texto en iris.data, ver nota abajo
ruta = "/Kmom/Home/analisis_numerico2/guia_5/iris.data"

datos = np.loadtxt(ruta, delimiter=",")

U, S, Vt = iris_svd(datos)
print("Valores singulares:", S)

# NOTA: si la ultima columna es texto (Iris-setosa, ...), reemplazar la lectura por:
# crudo = np.genfromtxt(ruta, delimiter=",", dtype=str)
# X = crudo[:, :-1].astype(float)
# nombres = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
# clases = np.array([nombres.index(c) for c in crudo[:, -1]])
# y armar datos = np.column_stack((X, clases))