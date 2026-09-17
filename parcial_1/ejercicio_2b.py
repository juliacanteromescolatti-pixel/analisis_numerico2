import numpy as np

#FUNCIONES AUXILIARES:

#MI FUNCION DE CHOLESKY DEL PRACTICO 1, EJRICIO 5:
def cholesky(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    G = np.zeros_like(A)

    for i in range(n):
        if A[i, i] <= 0:
            raise ValueError("La matriz no es definida positiva.")
        G[i, i] = np.sqrt(A[i, i])
        if i == n - 1:
            break
        G[i, i+1:] = A[i, i+1:] / G[i, i]
        G[i+1:, i] = G[i, i+1:]
        G_iJ = G[i, i+1:]
        A[i+1:, i+1:] -= np.outer(G_iJ, G_iJ)

    return np.tril(G)

#=================================================================================================================

#MIS FUNCIONES DE SUTITUCION DEL PRACTICO 1, EJRICIO 1:
def soltrinf_f(A, b):
    n = len(b)
    if np.any(np.diag(A) == 0):
        print("Error: la matriz es singular")
        return None
    x = np.zeros(n)
    for i in range(n):
        suma = 0
        for j in range(i):
            suma += A[i, j] * x[j]
        x[i] = (b[i] - suma) / A[i, i]
    return x


def soltrsup_f(A, b):
    n = len(b)
    if np.any(np.diag(A) == 0):
        print("Error: la matriz es singular")
        return None
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * x[j]
        x[i] = (b[i] - suma) / A[i, i]
    return x

#==========================================================================================================

#USAREMOS ESTAS TRES FUNCIONES PARA RESOLVER EL SITEMA AX=B SOLICITADO:
def resolver_cholesky(A, b):
    """
    Resuelve Ax = b via A = G G^T:
      1) G y = b con sustitucion hacia adelante (G es triang. inferior)
      2) G^T x = y con sustitucion hacia atras (G^T es triang. superior)
    """
    G = cholesky(A)          # G triangular inferior
    y = soltrinf_f(G, b)     # resuelve G y = b
    x = soltrsup_f(G.T, y)   # resuelve G^T x = y  (le paso G.T, que es triang. superior)
    return x, G
#RETORNO X Y G PUES ES LO QUE ME IMPORTA, LA SOLUCION (X) Y LA MATRIZ DE CHOLESKY (G).

#CARGAMOS LAS MATRICES DEL EJERICIO:
A = np.array([[3, 0,  0,  0,  1], [0, 9,  0,  0,  1], [0, 0, 27,  0,  1], [0, 0,  0, 81,  1], [1, 1,  1,  1, 243]], dtype=float)

b = np.array([4, 10, 28, 82, 247], dtype=float)

x, G = resolver_cholesky(A, b)

print("G =")
print(np.round(G, 4))
print("\nSolucion x =", x)
print("¿Ax == b?   ", np.allclose(A @ x, b))
print("Comparacion con np.linalg.solve:", np.allclose(x, np.linalg.solve(A, b)))