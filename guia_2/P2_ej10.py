# quiero a los ejercicios del algoritmo 5 y 6, agregarles permutacion,
# y testearlos
import numpy as np

def egaussp(A, b):
    n = len(b)
    # Copias flotantes para no modificar los datos originales
    U = A.copy().astype(float)
    y = b.copy().astype(float).reshape(-1, 1)

    for k in range(n - 1):
        # Buscamos el índice del máximo en la columna k (desde la fila k en adelante)
        # np.argmax nos da la posición relativa, sumamos 'k' para tener la fila real
        l = np.argmax(abs(U[k:n, k])) + k
        
        # Si el elemento máximo es cero, la matriz es singular
        if abs(U[l, k]) < 1e-12:
            raise ValueError("El pivote es cero. La matriz es singular.")

        # Si el máximo no está en la fila k, intercambiamos filas
        if l != k:
            # Intercambiamos filas tanto en U como en el vector y
            U[[k, l]] = U[[l, k]]
            y[[k, l]] = y[[l, k]]
            print(f"Iteración {k}: Se intercambió fila {k} con fila {l}")

        # Eliminación Gaussiana en el resto de las filas
        for i in range(k + 1, n):
            vi = U[i, k] / U[k, k]
            
            U[i, k] = 0.0  # Hacemos cero el elemento debajo del pivote
            U[i, k+1:n] = U[i, k+1:n] - vi * U[k, k+1:n]
            y[i, 0] = y[i, 0] - vi * y[k, 0]

    return U, y.flatten()

# =======================================================================
# TESTEO DE GAUSS
# =======================================================================
print("MATRIZ DE PRUEBA PARA GAUSS")
A = np.array([[0., 2, 1],
              [4., -1, 3],
              [2.,  8, -2]])
b = np.array([5., 11., 6.])

print("\nMatriz A:\n", A)
print("\nVector b:", b)

U, y = egaussp(A, b)

print("\n====================================================================")
print("SOLUCION")
print("La matriz U es:\n", U)
print("\nEl vector y es:", y)
print("====================================================================")

"""Convertís el vector b en una matriz columna para poder hacer el intercambio de filas
de forma segura y sin errores de sintaxis en NumPy.
Hacés toda la eliminación Gaussiana trabajando con esa estructura.
Al final, como el usuario y el resto del programa esperan un vector normal (unidimensional),
usás .flatten() para desarmar esa columna y devolverlo como el vector original."""

def dlup(A):
    n = len(A)
    # Inicializamos L como identidad, P como identidad y U como copia de A
    L = np.eye(n)
    P = np.eye(n)
    U = A.copy().astype(float)
    
    for k in range(n - 1):
        # Buscamos el índice del máximo en la columna k
        l = np.argmax(abs(U[k:n, k])) + k
        
        if abs(U[l, k]) < 1e-12:
            raise ValueError("La matriz es singular, no se puede realizar LU.")
            
        # Si el máximo no está en la diagonal, permutamos
        if l != k:
            U[[k, l]] = U[[l, k]]
            P[[k, l]] = P[[l, k]]
            # IMPORTANTE: Intercambiar también las filas previas de L (los multiplicadores ya calculados)
            if k > 0:
                L[[k, l], :k] = L[[l, k], :k]
        
        # Calculamos multiplicadores y actualizamos la submatriz U
        for i in range(k + 1, n):
            vi = U[i, k] / U[k, k]
            L[i, k] = vi       # Guardamos el multiplicador en L
            U[i, k] = 0.0      # Hacemos cero en U
            U[i, k+1:n] = U[i, k+1:n] - vi * U[k, k+1:n]
            
    return P, L, U

# =======================================================================
# TESTEO DE LU
# =======================================================================
print("\n\nMATRIZ DE PRUEBA PARA LU")
P, L, U_lu = dlup(A)

print("\nMatriz P (Permutación):\n", P)
print("\nMatriz L (Triangular Inferior):\n", L)
print("\nMatriz U (Triangular Superior):\n", U_lu)

print("\nVerificación: ¿P @ A == L @ U?")
print("P @ A:\n", P @ A)
print("L @ U:\n", L @ U_lu)