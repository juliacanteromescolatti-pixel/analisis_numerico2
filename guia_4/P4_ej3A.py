import numpy as np

def sign(X: float):
    if X >= 0:
        return 1
    else:
        return -1

def rotacion_givens_bloques(X1: float, X2: float):
    # queremos reducir cada elemento a cero
    if abs(X1) + abs(X2) == 0:
        c = 1.0
        s = 0.0
        return c, s
    elif abs(X2) > abs(X1):
        t = -X1/X2
        s = -sign(X2)/np.sqrt(1+t**2)
        c = t*s
        return c, s
    else:
        t = -X2/X1
        c = sign(X1)/np.sqrt(1+t**2)
        s = t*c
        return c, s

#==================================================================

print("Prueba del Algoritmo de givens con rotacion")
print()
A = np.array([[3.0, 5.0], [4.0, 2.0]])
print("La matriz original A es: ")
print(A)
print()
c, s = rotacion_givens_bloques(A[0,0], A[1,0]) # Tomamos la primer columna
# Construimos la matriz G
G = np.array([[c, -s], [s, c]])
print("La matriz G es: ")
print(G)
print()
# Multiplicamos G*A
R = np.dot(G, A)
print("La matriz R es: ")
print(R)
print()


"""
RESULTADO ESPERADO
La matriz resultado R deberia dar:
[[5. 4.6]
[0. -2.8]]
"""

"""
ARRAY = estructura de datos que almacena una coleccion de elementos
del mismo tipo en un orden secuencial.
-La funcion de numpy VSTACK apila los arrays de entrada en secuencia vertical.
-La funcion de numpy DOT realiza el producto de dos matrices.
"""
print("==========================================================================================================")

"""DESCOMP QR"""
"""INCISO A """
print("INCISO A")

# ENTRADA: MATRIZ A (m*n)
# SALIDA: MATRIZ Q Y R

import numpy as np

def qr_givens(A: np.ndarray):
    """Descomposición QR por Rotaciones de Givens siguiendo el pseudocódigo exacto."""
    m, n = A.shape # Calculo los tamaños de las filasy col
    R = A.copy().astype(float)  # A evoluciona directamente para terminar siendo R
    Q = np.eye(m, dtype=float)

    p = min(m - 1, n)

    # Nota: Se ajustan los rangos a indexación 0 de Python
    for j in range(p):
        for i in range(j + 1, m):
            if R[i, j] != 0:

                # c, s = rotacion_givens(a_jj, a_ij)
                c, s = rotacion_givens_bloques(R[j, j], R[i, j])

                # G = [[c, -s], [s, c]]
                G = np.array([[c, -s], [s, c]])

                # R_I,J <- G @ R_I,J
                R[[j, i], j:] = G @ R[[j, i], j:] # Esto nos devuelve una mattriz del mismo tamaño, actualiza por derecha

                # Q_*,I <- Q_*,I @ G^T
                Q[:,[j, i]]  = Q[:, [j, i]] @ G.T

    # Condición especial del pseudocódigo: Si m <= n y a_mm < 0
    # (Ajustado a índice m-1 en Python)
    if m <= n and R[m - 1, m - 1] < 0:
        R[m - 1, m-1:] = -R[m - 1, m-1:]
        Q[:, m - 1] = -Q[:, m - 1]
    return Q, R


# --- Verificación ---
A = np.random.rand(4, 5)
Q, R = qr_givens(A)
print("La matriz original A es: ")
print(A)
print()
print("La matriz Q es:")
print(Q)
print()
print("La matriz R es:")
print(R)
print()
print("La diferencia entre A y Q*R es:")
print(A - Q @ R)
