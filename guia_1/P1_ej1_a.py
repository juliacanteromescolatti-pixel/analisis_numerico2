#INCISO A
import numpy as np

# TRIANG INF POR FILAS
def soltrinf_f(A, b):
	n = len(b)

#Verifico la singularidad:
	if np.any(np.diag(A) == 0): # el np.any hace que si se cumple que A es diagonl lo imprime(recorre de arriba hacia abajo)
		print("Error: la matriz es singular")
		return None

	x = np.zeros(n)

	for i in range(n):
		suma = 0
		for j in range(i):
			suma += A[i,j]* x[j]

		x[i] = (b[i]-suma)/ A[i,i]

	return x

# MATRIZ DE PRUEBA
B = np.array([[1, 0, 0, 0], [2, 1, 0, 0], [1, 3, 2, 0], [4, 1, 2, 1]])

b_ = np.array([1, 4, 11, 12])

x_ = soltrinf_f(B, b_)
print("Solucion para una matriz triang inf por filas: ", x_)
print()
"""SOLUCION ESPERADA: [[1], [2], [2], [2]]"""

# TRIANG INF POR FILAS POR COLUMNAS
def soltrinf_c(A, b):
    n = len(b)

    # Verifico la singularidad:
    if np.any(np.diag(A) == 0):
        print("Error: la matriz es singular")
        return None

    x = np.zeros(n)
    # Hacemos una copia decimal de b para restarle los efectos de las columnas
    b_actualizado = b.astype(float) # Como restamos terminos similares para actualizar el b entonces lo pasamos a float

    # Recorremos por columnas de izquierda a derecha (0 hasta n-1)
    for j in range(n):
        # 1. Despejamos la incógnita actual usando el b modificado
        x[j] = b_actualizado[j] / A[j, j]

        # 2. Restamos el efecto de esta columna en las filas de abajo
        if j < n - 1:
            # A[j+1:, j] toma toda la columna j por debajo de la diagonal
            b_actualizado[j+1:] -= A[j+1:, j] * x[j]

    return x

# MATRIZ DE PRUEBA
# Matriz triangular inferior (A)
A = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 4.0, 2.0]])

# Vector de términos independientes (b)
b = np.array([4.0, 11.0, 20.0])

# Llamada a tu función
resultado = soltrinf_c(A, b)
print("Solucion para una matriz triang inf por columnas: ", resultado)



