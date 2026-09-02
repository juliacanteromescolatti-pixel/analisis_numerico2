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
from guia_1.P1_ej1_b import soltrsup_f


# Del Práctico 2 (Ej. 10): Descomposición LU con pivoteo parcial (PA = LU)
from P2_ej10 import dlup

#EJERCICIO 12: Cálculo de la Inversa usando tus funciones

def inv_lu(A):
    """
    Calcula la inversa de una matriz A utilizando descomposición LU 
    con permutaciones resolviendo n sistemas lineales funciones previas.
    """
    n = A.shape[0]
    
    # Paso 1: Factorizar la matriz usando tu función dlup (Práctico 2, Ej 10)
    A_lu, P = dlup(A)
    L = np.tril(A_lu, -1) + np.eye(n) #Toma la parte estrictamente por debajo de la diagonal principal.
    U = np.triu(A_lu) #En u no solo tomo la parte de arriba de la diagonal, sino tambien la diagonal incluida.
    
    # Inicializar matriz identidad e inversa vacía
    I = np.eye(n)
    A_inv = np.zeros((n, n))
    
    # Paso 2: Resolver un sistema para cada columna de la identidad
    for j in range(n):
        e_j = I[:, j]  # j-ésimo vector canónico
        
        # Aplicamos la permutación al vector de términos independientes: P * e_j
        b_perm = np.dot(P, e_j)
        
        # Sustitución hacia adelante (Práctico 1, Ej 1a): L y = P e_j
        y = soltrinf_f(L, b_perm)
        
        # Sustitución hacia atrás (Práctico 1, Ej 1b): U x = y
        x = soltrsup_f(U, y)
        
        # Guardamos el vector resultante en la j-ésima columna de la inversa
        A_inv[:, j] = x
        
    return A_inv


#TESTEO (Con la matriz A del Ejercicio 11)
print("--- TESTEO EJERCICIO 11 ---")
# Matriz A definida en el enunciado del Ejercicio 11
A = np.array([
        [2, 10,  8,  8,  6],
        [1,  4, -2,  4, -1],
        [0,  2,  3,  2,  1],
        [3,  8,  3, 10,  9],
        [1,  4,  1,  2,  1]], dtype=float)

print("Matriz original A del Ejercicio 11:")
print(A)

# Ejecutamos tu función para obtener la inversa
A_inv_resultado = inv_lu(A)

print("Matriz Inversa calculada con inv_lu:")
print(np.round(A_inv_resultado, 4))

# Ese , 4 es el segundo argumento que recibe la función np.round(). 
#Sirve para indicarle a Python cuántos decimales quieres que conserve al hacer el redondeo.
    
# Verificación automática comparando con NumPy
print("¿El resultado coincide exactamente con np.linalg.inv?:")
print(np.allclose(A_inv_resultado, np.linalg.inv(A)))

"""np.allclose: Sirve para comparar si dos matrices son numéricamente "iguales", 
tolerando los pequeñísimos errores de precisión decimal que genera la computadora.
np.round: Sirve para redondear los elementos de una matriz al número de decimales 
que quieros."""