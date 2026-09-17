import numpy as np

#FUNCIONES AUXILIARES:

#MI FUNCION PARA RESOLVER LU DEL PRACTICO 2, EJRICIO 6:
def dlu(A):
    n = A.shape[0]
    U = A.copy()
    for k in range(n):
        U[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - np.outer(U[k+1:, k], U[k, k+1:])

    L = np.tril(U, -1) + np.eye(n)
    U = np.triu(U)

    return U, L

#===============================================================================================================

#USAREMOS MI FUNCION LU PARA RESOLVER LO SOLICITADO EN ELJERCICIO 4B:
def lu_4b(a, b, c):
    """
    A = [[a, a, a],
         [a, b, b],
         [a, b, c]]

    Condiciones para pivots no nulos (deducidas haciendo la
    eliminacion gaussiana a mano en el punto 4a):
      pivot 1 = a       -> a != 0
      pivot 2 = b - a   -> b != a
      pivot 3 = c - b   -> c != b
    """
    if a == 0 or b == a or c == b:
        print("no se satisfacen las condiciones")
        return None

    A = np.array([[a, a, a],
                  [a, b, b],
                  [a, b, c]], dtype=float)

    U, L = dlu(A)
    return L, U


# Caso que SI cumple las condiciones
L, U = lu_4b(2, 5, 9)
print("L =", L)
print("U =", U)
print("¿L@U == A?", np.allclose(L @ U, np.array([[2,2,2],[2,5,5],[2,5,9]], dtype=float)))

print()
print("=============================================================================================")

# Caso que NO cumple las condiciones (a == 0)
lu_4b(0, 5, 9)
print()
print("=============================================================================================")

# Caso que NO cumple las condiciones (b == a)
lu_4b(3, 3, 9)