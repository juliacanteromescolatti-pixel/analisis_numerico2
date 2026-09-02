#Nos pide eliminacion Gaussiana con pivoteo parial para resolver Ax=b
#A diferencia del 10 aca debo resolver un sistema completo, donde en una parte usare lo definido en el ejrcicio 10.
import numpy as np
# Importamos egaussp desde tu archivo del ejercicio 10
from P2_ej10 import egaussp
# Importamos tus funciones de sustitución
import sys
import os

# Agrega la ruta de la carpeta que contiene a 'guia_1'
# Reemplaza '/home/kmom/analisis_numerico2/' por tu ruta base si es otra
sys.path.append('/home/kmom/analisis_numerico2/')

# Ahora importas la función usando puntos (.) para las carpetas
from guia_1.P1_ej1_b import soltrsup_f
from guia_1.P1_ej1_b import soltrsup_c

def sol_egauss(A, b):
    """
    Utiliza eliminación Gaussiana con pivoteo parcial para resolver Ax = b.
    Reutiliza las funciones importadas externamente.
    """
    # 1. Fase de Eliminación (reutiliza tu código importado)
    U, y = egaussp(A, b)
    
    # 2. Fase de Sustitución (podés usar soltrsup_f o soltrsup_c, dan lo mismo)
    x = soltrsup_c(U, y)
    
    return x

# =======================================================================
# TESTEO DEL EJERCICIO 11
# =======================================================================
print("--- TESTEO EJERCICIO 11 ---")

# Datos del enunciado
A = np.array(
    [
        [2, 1, 0, 4, -2],
        [8, 8, 2, 4, -1],
        [6, 5, 3, 1, 4],
        [2, 2, 2, 2, 1],
        [1, 1, 1, 1, 15],
    ]
)

b1 = np.array([10, 52, 50, 12, 12])
b2 = np.array([8, 50, 48, 12, 12])

# Resolver y testear para b1
x1 = sol_egauss(A, b1)
print("Solución para b1:")
print(x1)
print("Verificación Ax1 == b1:", np.allclose(A @ x1, b1))
print()

# Resolver y testear para b2
x2 = sol_egauss(A, b2)
print("Solución para b2:")
print(x2)
print("Verificación Ax2 == b2:", np.allclose(A @ x2, b2))