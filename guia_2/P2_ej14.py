import numpy as np
import time
from P2_ej10 import dlup 

def det_lu(A):
    """
    Calcula el determinante de una matriz A utilizando su 
    descomposición LU con permutaciones.
    
    Explicación matemática del proceso paso a paso:
    """
    # PASO 1: Obtenemos la descomposición PA = LU ya programada en el Ejercicio 10.
    # Recibimos las tres salidas. Llamamos 'P_salida' a la primera para analizarla.
    # Según la teoría, la descomposición devuelve P, L, U o L, U, P.
    salida1, salida2, salida3 = dlup(A)
    
    # Para saber cuál es U (la triangular superior), buscamos cuál tiene ceros abajo.
    # Por descarte, identificamos las matrices correctamente:
    if np.allclose(salida1, np.triu(salida1)) == False and np.allclose(salida2, np.triu(salida2)) == False:
        # Si la tercera salida es U (caso L, U, P)
        U = salida2
        P_matriz = salida3
    else:
        # Si la tercera salida es U (caso P, L, U)
        U = salida3
        P_matriz = salida1
    
    # PASO 2: Calculamos el determinante de la matriz triangular superior U.
    # Por propiedad matemática, el determinante de cualquier matriz triangular es 
    # igual al producto de los elementos que están en su diagonal principal.
    # - np.diagonal(U) extrae los números de la diagonal.
    # - np.prod() los multiplica todos entre sí.
    det_u = np.prod(np.diagonal(U))
    
    # PASO 3: 
    # Calculamos el determinante de la matriz de permutación para obtener el signo (1 o -1).
    # Si tu salida3 era un vector en vez de una matriz, obtenemos su determinante convirtiéndola.
    if P_matriz.ndim == 1:
        n = len(P_matriz)
        I = np.eye(n)
        P_matriz = I[P_matriz.astype(int)]
        
    signo = np.linalg.det(P_matriz)
    """"Este bloque analiza si tu código del Ejercicio 10 arrojó una matriz o una lista de posiciones. 
    Si arrojó una lista, el código fabrica la matriz correspondiente en milisegundos para poder 
    extraer el signo correcto (1 o -1) sin que se rompa por incompatibilidad de tipos de datos."""

    # PASO 4: Devolver el determinante final de la matriz A.
    # Como el determinante de L es siempre 1 (tiene unos en la diagonal), el determinante 
    # final de A es simplemente el determinante de U multiplicado por el signo obtenido.
    return signo * det_u


#Matriz de prueba del Ejercicio 11
A = np.array([
        [2, 10,  8,  8,  6],
        [1,  4, -2,  4, -1],
        [0,  2,  3,  2,  1],
        [3,  8,  3, 10,  9],
        [1,  4,  1,  2,  1]
    ], dtype=float)

print("=== COMPARACIÓN DE RESULTADOS ===")

# Medición con tu función det_lu
t0 = time.perf_counter()  # 1. Enciendes el cronómetro antes de empezar (guarda el tiempo en t0)
mi_det = det_lu(A)        # 2. Tu computadora ejecuta el algoritmo LU y calcula el determinante
t1 = time.perf_counter()  # 3. Detienes el cronómetro apenas termina (guarda el tiempo en t1)


# Medición con la función nativa de NumPy
t2 = time.perf_counter()  # 1. Vuelves a encender el cronómetro (guarda el tiempo en t2)
np_det = np.linalg.det(A) # 2. NumPy calcula el determinante de forma interna y automática
t3 = time.perf_counter()  # 3. Detienes el cronómetro (guarda el tiempo en t3)

print(f"Determinante det_lu: {mi_det:.4f} (Tiempo: {t1-t0:.6f}s)")
print(f"Determinante NumPy:  {np_det:.4f} (Tiempo: {t3-t2:.6f}s)")

"""Las partes {mi_det:.4f} y {np_det:.4f} muestran los determinantes con 4 decimales para 
comprobar que den exactamente lo mismo (el resultado matemático idéntico).
Las operaciones {t1-t0:.6f}s y {t3-t2:.6f}s hacen la resta del cronómetro en vivo y
muestran la cantidad de segundos con 6 decimales."""

"""np.linalg.det" es la función oficial y nativa de la librería NumPy encargada de 
calcular el determinante de una matriz cuadrada de forma automática."""






