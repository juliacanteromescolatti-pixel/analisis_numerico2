
import numpy as np

def egauss(A, b, n):
    A_gauss = A.copy().astype(float)
    # Convertimos 'b' a un vector columna (n, 1) de tipo flotante
    b_gauss = b.copy().astype(float).reshape(-1, 1)

    for k in range(n - 1):
        if A_gauss[k, k] == 0:
            return None, None

        for i in range(k + 1, n):
            m = A_gauss[i, k] / A_gauss[k, k]

            # Actualizamos la fila i de A_gauss desde la columna k en adelante
            for j in range(k, n):
                A_gauss[i, j] = A_gauss[i, j] - m * A_gauss[k, j]

            # Hacemos 0 explícito el elemento debajo del pivote por precisión
            A_gauss[i, k] = 0.0

            # Actualizamos el vector b
            b_gauss[i, 0] = b_gauss[i, 0] - m * b_gauss[k, 0]

    U = A_gauss
    y = b_gauss
    return U, y

# FUNCION DE PRUEBA 1

A = np.array([[2., 1, 1], [4, 3, 3], [8, 7, 9]])
b = np.array([1., 2, 3])

U, y = egauss(A, b, 3)

print("--- PRUEBA 1: egauss ---")
print("U =\n", U)
print("y =\n", y)

"""
RESPUESTA ESPERADA:
U = [[2, 1, 1], [0, 1, 1], [0, 0, 2]]
y = [[1], [0], [-1]]
"""

# EXTRA

def soltrsup(A, b):
    n = len(b)

    # Verifico la singularidad:
    if np.any(np.diag(A) == 0):
        print("Error: la matriz es singular")
        return None

    x = np.zeros((n, 1))

    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += A[i, j] * x[j, 0]

        x[i, 0] = (b[i, 0] - suma) / A[i, i]

    return x

def soleg(A, b):
    U, y = egauss(A, b, 3)
    x = soltrsup(U, y)
    return x

# FUNCION DE PRUEBA 2

A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
b = np.array([[1], [2], [3]])

x = soleg(A, b)

print("\n--- PRUEBA 2: soleg ---")
print("x =\n", x)

"""RESPUESTA ESPERADA: [[1], [-1], [0]]"""