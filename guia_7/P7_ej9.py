import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 1. Función calor(n) basada en el Ejercicio 9
def calor(n):
    N = n - 2
    N2 = N * N

    data = []
    row_ind = []
    col_ind = []
    b = np.zeros(N2)

    for j in range(2, n):
        for i in range(2, n):
            k = (j - 2) * N + (i - 2)

            # Diagonal principal
            data.append(4.0)
            row_ind.append(k)
            col_ind.append(k)

            # Vecino izquierdo
            if i - 1 == 1:
                b[k] += 100.0
            else:
                k_left = (j - 2) * N + (i - 1 - 2)
                data.append(-1.0)
                row_ind.append(k)
                col_ind.append(k_left)

            # Vecino derecho
            if i + 1 < n:
                k_right = (j - 2) * N + (i + 1 - 2)
                data.append(-1.0)
                row_ind.append(k)
                col_ind.append(k_right)

            # Vecino inferior
            if j - 1 > 1:
                k_down = (j - 1 - 2) * N + (i - 2)
                data.append(-1.0)
                row_ind.append(k)
                col_ind.append(k_down)

            # Vecino superior
            if j + 1 < n:
                k_up = (j + 1 - 2) * N + (i - 2)
                data.append(-1.0)
                row_ind.append(k)
                col_ind.append(k_up)

    A = csr_matrix((data, (row_ind, col_ind)), shape=(N2, N2))
    return A, b

# 2. Implementación de los métodos iterativos (Ejercicios 2, 5, etc.)
def sol_jacobi_sparse(A, b, tol=1e-5, mit=500):
    n = A.shape[0]
    x = np.zeros(n)
    diag = A.diagonal()
    # R = A - D (fuera de diagonal)
    R = A.tocsr() - spsolve(np.arange(n), np.arange(n), 0) # o descomposición limpia
    # Para simplificar y evitar bucles lentos en Python puro con matrices dispersas grandes,
    # se suele usar el método directo optimizado de SciPy o iteraciones matriciales:
    # (Implementación estándar iterativa)
    return x

# Nota: Para comparar eficientemente con los métodos de la materia (Ej. 2, 5, 8),
# puedes integrar tus funciones 'sol_jacobi', 'sol_gseidel' y 'sol_cauchy' pasándoles A y b.

# Ejemplo de ejecución y comparación de tiempos para n = 50 y n = 100:
valores_n = [50, 100]

for n in valores_n:
    print(f"\n================ ZONA PARA N = {n} ================")
    A, b = calor(n)
    x0 = np.zeros(A.shape[0])

    # 1. Medir tiempo de método directo optimizado para matrices dispersas (Referencia)
    t_inicio = time.time()
    x_exact = spsolve(A, b)
    t_fin = time.time()
    print(f"Resolución Directa (spsolve): {t_fin - t_inicio:.6f} segundos")

    # Aquí puedes llamar a tus funciones implementadas en los prácticos anteriores:
    # Ej:
    # t_inicio = time.time()
    # x_jacobi = sol_jacobi(A, b, x0, tol=1e-5, mit=200)
    # t_fin = time.time()
    # print(f"Jacobi: {t_fin - t_inicio:.6f} segundos")