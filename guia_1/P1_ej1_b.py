#INCISO B
import numpy as np

# TRIANG SUP POR FILAS
def soltrsup_f(A, b):
    n = len(b)

    # Verifico la singularidad:
    if np.any(np.diag(A) == 0):
        print("Error: la matriz es singular")
        return None

    x = np.zeros(n)

    # Recorremos las filas desde la última hasta la primera
    for i in range(n-1, -1, -1): 
        suma = 0
        for j in range(i+1, n):
            suma += A[i, j] * x[j]

        x[i] = (b[i] - suma) / A[i, i]

    return x


# TRIANG SUP POR COLUMNAS
def soltrsup_c(A, b):
    n = len(b)

    # Verifico la singularidad:
    if np.any(np.diag(A) == 0):
        print("Error: la matriz es singular")
        return None

    x = np.zeros(n)
    # Hacemos una copia decimal de b y la aplanamos a 1D
    b_actualizado = b.astype(float).flatten() 

    # Recorremos por columnas de derecha a izquierda (n-1 hasta 0)
    for j in range(n-1, -1, -1):
        # 1. Despejamos la incógnita actual usando el b modificado
        x[j] = b_actualizado[j] / A[j, j]

        # 2. Restamos el efecto de esta columna en las filas de ARRIBA
        if j > 0:
            # A[:j, j] toma toda la columna j por encima de la diagonal
            b_actualizado[:j] -= A[:j, j] * x[j]

    return x  # ¡Importante regresar el resultado!


#MATRIZ DE PRUEBA PARA VERIFICAR
A = np.array([[1, 2, 3, 1], [0, 2, 1, 4], [0, 0, 3, 2], [0, 0, 0, 5]])
b = np.array([[16], [15], [13], [10]])

x_filas = soltrsup_f(A, b)
x_columnas = soltrsup_c(A, b)

print("Solución por FILAS:   ", x_filas)
print("Solución por COLUMNAS:", x_columnas)
"""SOLUCION ESPERADA: [1. 2. 3. 2.]"""


"""
    for j in range(n-1, -1, -1):
El primer termino es donde empezamos, entonces n-1 es porque empezamos en la columna n.
El segundo termino es donde terminamos, entonces -1 es porque queremos parar en la columna 0.
El tercer termino es como nos movemso, entonces -1 es porque nos movemos de uno en uno hacia atras.
    """