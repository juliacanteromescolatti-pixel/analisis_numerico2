import numpy as np
"""Definimos las matrices A y B"""
A = np.array([[1, 0, 4, 2], [2, -7, 1, 0], [0, 0, -3, 1], [-4, 1, 0, 2]])
B = np.array([[1, 2, 0, -1], [2, 0, -3, 0], [-1, 2, 0, -1], [1, 0, 1, 5]])
C = np.array([[-1, 10, 2, 5], [-13, 6, 21, -3], [4, -6, 1, 8], [0, -8, -1, 14]])
"""
Filas 0:2 es la fila 0 y la fila 2
Fila 2:4 es la fila dos hasta la fila 4
Para las columnas es el mismo razonamiento
Partimos las matrices
"""
A11 = A[0:2, 0:3]
A12 = A[0:2, 3:4]
A21 = A[2:4, 0:3]
A22 = A[2:4, 3:4]

B11 = B[0:3, 0:2]
B12 = B[0:3, 2:4]
B21 = B[3:4, 0:2]
B22 = B[3:4, 2:4]

"""Calculamos la matriz C"""
C11 = A11 @ B11 + A12 @ B21
C12 = A11 @ B12 + A12 @ B22
C21 = A21 @ B11 + A22 @ B21
C22 = A21 @ B12 + A22 @ B22

"""Imprimimos matrices"""
print("El bloque C11 es: ")
print(C11)
print()
print("El bloque C12 es: ")
print(C12)
print()
print("El bloque C21 es: ")
print(C21)
print()
print("El bloque C22 es: ")
print(C22)
print()
