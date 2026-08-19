import numpy as np

# Traemos las funciones del archivo inciso_a
from P1_ej1_a import soltrinf_f, soltrinf_c

# Traemos las funciones del archivo inciso_b
from P1_ej1_b import soltrsup_f, soltrsup_c


#PRUEBAS DE MATRICES TRIANGULARES INFERIORES
print("--- MATRICES TRIANGULARES INFERIORES ---")
A1 = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -2, 2]])
B1 = np.array([0, 0, 1, 1])

A2 = np.array([[2, 0, 0, 0], [-1, 2, 0, 0], [3, 1, -1, 0], [4, 1, -3, 3]])
B2 = np.array([2, 3, 2, 9])

# Usando Filas
print("Matriz 1 por filas: ", soltrinf_f(A1, B1))
print("Matriz 2 por filas: ", soltrinf_f(A2, B2))

# Usando Columnas
print("Matriz 1 por col:   ", soltrinf_c(A1, B1))
print("Matriz 2 por col:   ", soltrinf_c(A2, B2))
print()


#PRUEBAS DE MATRICES TRIANGULARES SUPERIORES
print("--- MATRICES TRIANGULARES SUPERIORES ---")
A3 = np.array([[9, 2, 4], [ 0, -6, 3], [0, 0, 5]])
B3 = np.array([18, -2, 7])

A4 = np.array([[1, 2, -1, 1], [0, 1, 0, -1], [0, 0, -1, 4], [0, 0, 0, 1]])
B4 = np.array([2, -1, 0, 0])

# Usando Filas
print("Matriz 3 por filas: ", soltrsup_f(A3, B3))
print("Matriz 4 por filas: ", soltrsup_f(A4, B4))

# Usando Columnas (¡Ya no va a dar None!)
print("Matriz 3 por col:   ", soltrsup_c(A3, B3))
print("Matriz 4 por col:   ", soltrsup_c(A4, B4))