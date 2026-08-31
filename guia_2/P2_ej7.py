"""En la eliminación Gaussiana tradicional necesitas dos bucles anidados para modificar las 
filas inferiores. Aquí, para cada fila (k), el único elemento debajo del pivote en su misma 
columna está en la fila inmediata siguiente (k+1), excepto en la primera columna, que 
también tiene un elemento en la última fila (n).Además, al modificar las filas, 
solo cambian de valor unas pocas posiciones"""

#PSEUDOCODIGO:
"""
Algoritmo eliminación_gaussiana_estructurada(A, b):
    n = dimensión de A
    
    # --- PASO 1: Eliminar el elemento de la primera columna, última fila (a_n1) ---
    factor = A[n, 1] / A[1, 1]
    A[n, 1] = 0
    A[n, 2] = A[n, 2] - factor * A[1, 2]
    A[n, n] = A[n, n] - factor * A[1, n]
    b[n] = b[n] - factor * b[1]
    
    # --- PASO 2: Eliminación en cascada para el resto de las filas ---
    Para k desde 2 hasta n-1:
        # Solo la fila de abajo (k+1) tiene un elemento a eliminar en la columna k
        factor = A[k+1, k] / A[k, k]
        A[k+1, k] = 0
        
        # Modificamos los elementos de la fila k+1 que se ven afectados
        A[k+1, k+1] = A[k+1, k+1] - factor * A[k, k+1]
        
        # Si la fila k tiene un elemento en la última columna, afecta a la fila k+1
        Si A[k, n] != 0:
            A[k+1, n] = A[k+1, n] - factor * A[k, n]
            
        b[k+1] = b[k+1] - factor * b[k]
        
    Retornar A (que ahora es la matriz U triangular superior) y b (que ahora es y)
"""

import numpy as np

def egauss_estructurado(A, b):
    """
    Eliminación Gaussiana sin pivoteo optimizada para matrices 
    con estructura tridiagonal más las esquinas (a_1n) y (a_n1).
    """
    n = len(b)
    # Copiamos para no modificar los datos originales recibidos
    U = A.copy().astype(float)
    y = b.copy().astype(float)
    
    #PASO 1: Eliminar el elemento de la esquina inferior izquierda U[n-1, 0] ---
    if U[n-1, 0] != 0:
        factor = U[n-1, 0] / U[0, 0]
        U[n-1, 0] = 0
        U[n-1, 1] -= factor * U[0, 1]
        U[n-1, n-1] -= factor * U[0, n-1]
        y[n-1] -= factor * y[0]
        
    #PASO 2: Eliminación para el resto de las filas en la banda ---
    for k in range(1, n-1):
        # Eliminamos el elemento justo debajo de la diagonal: U[k+1, k]
        if U[k+1, k] != 0:
            factor = U[k+1, k] / U[k, k]
            U[k+1, k] = 0
            
            # Re値culamos el elemento de la diagonal de la fila siguiente
            U[k+1, k+1] -= factor * U[k, k+1]
            
            # Si el elemento en la última columna de la fila actual no es cero, afecta al de abajo
            U[k+1, n-1] -= factor * U[k, n-1]
            
            # Modificamos el vector del lado derecho
            y[k+1] -= factor * y[k]
            
    return U, y


#MATRIZ DE PRUEBA
# Creamos una matriz que cumpla exactamente con la estructura solicitada
A_prueba = np.array([
    [4.0, 1.0, 0.0, 0.0, 2.0],  # Esquina a_1n es 2.0
    [1.0, 4.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 4.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 4.0, 1.0],
    [3.0, 0.0, 0.0, 1.0, 4.0]   # Esquina a_n1 es 3.0
], dtype=float)

b_prueba = np.array([7.0, 6.0, 6.0, 6.0, 8.0], dtype=float)

print("=== TESTEO EJERCICIO 7 ===")
U_resultado, y_resultado = egauss_estructurado(A_prueba, b_prueba)

print("Matriz U resultante (Debe ser triangular superior):")
print(np.round(U_resultado, 4)) #EL 4 ES PARA LA CANTE DE DECIMALES
print("\nVector y resultante:")
print(np.round(y_resultado, 4))
