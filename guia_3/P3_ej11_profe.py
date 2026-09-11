import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 1º PRIMERO: Le enseñamos a Python dónde están tus carpetas
sys.path.append('/home/kmom/analisis_numerico2')
sys.path.append('/home/kmom/analisis_numerico2/guia_2')

# 2º SEGUNDO: Recién ahora hacemos la importación
from guia_2.P2_ej11 import sol_egauss



# 1. Obtener la ruta del directorio donde está guardado este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Definir rutas absolutas para evitar el FileNotFoundError
ruta_archivo_A = os.path.join(script_dir, "A_dataset.txt")
ruta_archivo_b = os.path.join(script_dir, "b_dataset.txt")

# 3. Cargar los datos usando NumPy
A = np.loadtxt(ruta_archivo_A)
b = np.loadtxt(ruta_archivo_b)

def matriz_per(A,b):

    n = A.shape[0]

    x_exact = sol_egauss(A,b)

    E = np.random.randn(n)

    delta_x = []
    delta_A = []

    for beta in range(1, 11):

        epsilon = 10**(-beta)

        A_tilde = A + epsilon*E

        x_tilde = sol_egauss(A_tilde, b)

        err_x = np.linalg.norm(x_tilde - x_exact, 2) / np.linalg.norm( x_exact, 2)

        err_A = np.linalg.norm(A_tilde - A, 2) / np.linalg.norm(A, 2)

        delta_x.append(err_x)
        delta_A.append(err_A)

        plt.loglog(delta_A, delta_x, 'o-', color='r')

    plt.xlabel("delta A")
    plt.ylabel("delta x")
    plt.title('Sensibilidad de la solución frente a perturbaciones en A')
    plt.show()

#matriz_per(A,b)
# La matriz de datos no es mal condicionada por lo que esta bien que la gráfica sea lineal
# voy a dejar un ejemplo de una matriz mal condicionada 

n = len(b)

A_mal_cond = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        A_mal_cond[i, j] = 1 / (i + j + 1)

matriz_per(A_mal_cond, b)