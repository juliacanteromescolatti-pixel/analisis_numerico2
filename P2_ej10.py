# quiero a los ejercicios del algoritmo 5 y 6, agregarles permutacion,
# y testearlos


# =======================================================================
# GAUSS CON PERMU
# =======================================================================

"""
ENTRADA: matriz A y vector b
SALIDA: matriz U y vector y de un sistema equivalente
"""

import numpy as np

def egauss(A, b):
    n = len(b)
    A_gauss = A.copy().astype(float)
    # Convertimos 'b' a un vector columna (n, 1) de tipo flotante
    b_gauss = b.copy().astype(float).reshape(-1, 1)

    for k in range(n - 1):
        l = 0
        for j in range (k, n+1):
          if abs(A_gauss[l, k]) != max(abs(A_gauss[j:, k])):
            l = l + 1
          else:
            print("necesitamos ", l, " iteraciones.")

            if l != k:
            # Intercambio las filas l y k y despues actualizo la matriz
                A_gauss[[l, k]] = A_gauss[[k, l]]
                A_gauss = A.copy().astype(float)
        for i in range(k + 1, n):
                A_gauss[i, k] = 0.0
                vi = A_gauss[i, k]/ A_gauss[k, k]
                A_gauss[i, k+1:n] = A_gauss[i, k+1:n] - vi * A_gauss[k, k+1:n]
                bi = b_gauss[i, 0] - vi * b_gauss[k, 0]

    U = A_gauss
    y = bi
    return U, y




# MATRIZ DE PRUEBA
print("MATRIZ DE PRUEBA PARA GAUSS")
print()
A = np.array([[0., 2, 1],
              [4., -1, 3],
              [2.,  8, -2]])
print(A)
print()
b = np.array([5.,11.,6.])
print("El vector b es: ")
print(b)
print()
sol = egauss(A, b)
print("====================================================================")
print()
print("SOLUCION")
print("La matriz U es: ")
print(sol[0])
print()
print("El vector y es: ")
print(sol[1])
print()
print("====================================================================")
print()



# =======================================================================
# LU CON PERMU
# =======================================================================

"""
ENTRADA: matriz A
SALIDA: matrices L, U, P (Tr inf, tr sup, permu respc) tq P*A = L*U
"""

import numpy as np

def LU_perm(A):
    n = len(A)
    P = np.eye(n)
    for k in range (n-1)

