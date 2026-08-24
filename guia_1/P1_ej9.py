#EL jercio 9 nos pide
import numpy as np

# Importamos las sustituciones desde el archivo del Ejercicio 1
from P1_ej1_a import soltrinf_f
from P1_ej1_b import soltrsup_f

# Importamos el algoritmo de Cholesky desde el archivo del Ejercicio 5
from P1_ej5 import cholesky

# ACTIVIDAD 9
# necesitamos Choles pq el sistema es triang inf
import numpy as np # Importamos numpy para usar la funcion transpouse en L
import P1_ej5
import P1_ej1_a, P1_ej1_b

def sol_defpos(A, b):
# PASO 1 calculamos la matriz L usando cholesky, pues pensamos a A como
# A = L*L_trans
    L = P1_ej5.cholesky(A)
    L_trans = np.transpose(L)
# PASO 2 resolver L*y = b
    y = P1_ej1_a.soltrinf_f(L, b)
# PASO 3 resolver L_trans*x = y
    x = P1_ej1_b.soltrsup_f(L_trans, y)
    return x

# MATRIZ DE PRUEBA
A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
b = np.array([1, 2, 3])

sol = sol_defpos(A, b)
print("la solucion es=", sol)