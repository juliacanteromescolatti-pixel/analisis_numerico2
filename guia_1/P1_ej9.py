#EL jercio 9 nos pide
import numpy as np

# Importamos las sustituciones desde el archivo del Ejercicio 1
from P1_ej1_a import soltrinf_f
from P1_ej1_b import soltrsup_f

# Importamos el algoritmo de Cholesky desde el archivo del Ejercicio 5
from P1_ej5 import cholesky

def sol_defpos(A, b):
    """
    Resuelve el sistema Ax = b para una matriz A definida positiva
    mediante una descomposición de Cholesky y la resolución de dos
    sistemas triangulares.
    """
    # 1. Obtenemos el factor triangular inferior G tal que A = G * G^T
    G = cholesky(A)
    
    # 2. Resolvemos el sistema triangular inferior: G * y = b
    y = soltrinf_f(G, b)
    
    # 3. Resolvemos el sistema triangular superior: G^T * x = y
    Gt = G.T
    x = soltrsup_f(Gt, y)
    
    return x



# MATRIZ DE PRUEBA

# Matriz de prueba (Matriz B del Ejercicio 8)
A_prueba = np.array([[4, 2, 6], [2, 2, 5], [6, 5, 29]])

b_prueba = np.array([6, 2, 6])

# Resolver con nuestra función limpia
x_calculada = sol_defpos(A_prueba, b_prueba)

print("EJERCICIO 9 COMPLETO")
print("Solución calculada con sol_defpos: ", x_calculada)