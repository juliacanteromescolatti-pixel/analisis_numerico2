import numpy as np
import sys
import os

# Agrega la ruta de la carpeta que contiene a 'guia_1'
# Reemplaza '/home/kmom/analisis_numerico2/' por tu ruta base si es otra
sys.path.append('/home/kmom/analisis_numerico2/')

# Ahora importas la función usando puntos (.) para las carpetas
# Del Práctico 1 (Ej. 1a): Sustitución hacia adelante para Ly = b
from guia_1.P1_ej1_a import soltrinf_f

# Del Práctico 1 (Ej. 1b): Sustitución hacia atrás para Ux = y
from guia_1.P1_ej1_b import soltrsup_f, soltrsup_c


# Del Práctico 2 (Ej. 10): 
from P2_ej10_profe import egaussp

def sol_egauss(A,b):
    U, y = egaussp(A,b)
    x = soltrsup_c(U, y)

    return x

A = np.array([[2., 10, 8, 8, 6],
              [1, 4, -2, 4, -1],
              [0, 2, 3, 2, 1],
              [3, 8, 3, 10, 9],
              [1, 4, 1, 2, 1]])

b_1 = np.array([52., 14, 12, 51, 15])
b_2 = np.array([50., 4, 12, 48, 12])

sol_1 = sol_egauss(A,b_1)

sol_2 = sol_egauss(A,b_2)

print(f'sol_2 = {sol_2}')

print(f'sol_1 = {sol_1}')

