import numpy as np

def arnoldi(A, v, m):
    A = np.array(A, dtype=float)
    n = A.shape[0] # Dimensión de la matriz A (n x n)
    v = np.array(v, dtype=float)

    if m > n:
        raise ValueError("m debe ser menor o igual que n")

    # Inicialización de las matrices de salida según el algoritmo
    V = np.zeros((n, m))
    H_bar = np.zeros((m + 1, m))

    # Paso 0: Si la norma del vector inicial es 0
    if np.linalg.norm(v, ord=2) == 0:
        return H_bar, V

    # Paso 1: Bucle principal para j = 1, ..., m (en Python de 0 a m-1)
    for j in range(m):
        norm_v = np.linalg.norm(v, ord=2)
        if norm_v == 0:
            break

        # Normalización y asignación de la columna j-ésima de V
        V[:, j] = v / norm_v

        # Primer producto con A
        v = A @ V[:, j]

        # 2. Ortogonalización (Bucle interno para i = 1, ..., j)
        for i in range(j + 1):  # En Python los índices van de 0 a j
            H_bar[i, j] = V[:, i].T @ v
            v = v - H_bar[i, j] * V[:, i]

        # 3. Comprobación y cálculo del elemento subdiagonal
        norm_v = np.linalg.norm(v, ord=2)
        if norm_v == 0:
            break
        else:
            H_bar[j + 1, j] = norm_v

    return H_bar, V

