#Nos pide relaizar el algoritmo de Cholesky
# ALGORITMO DE CHOLESKY
import numpy as np

def cholesky(A):
    """
    Realiza la descomposición de Cholesky A = G * G^T
    siguiendo la idea por bloques de la guía.
    Modifica una copia de la matriz in-place para mayor eficiencia.
    """
    # Convertimos la entrada a flotante y hacemos una copia para no alterar la original
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Aquí iremos guardando la matriz triangular inferior G
    G = np.zeros_like(A)

    for i in range(n):
        # El subíndice J representa el bloque desde i+1 hasta el final (n)

        # Paso 1: g_ii = sqrt(a_ii)
        if A[i, i] <= 0:
            raise ValueError("La matriz no es definida positiva.")

        G[i, i] = np.sqrt(A[i, i])

        # Si ya estamos en el último elemento, terminamos el bucle
        if i == n - 1:
            break

        # Paso 2: G_iJ = (1 / g_ii) * A_iJ
        # A[i, i+1:] representa la fila i desde la columna i+1 en adelante (A_iJ)
        G[i, i+1:] = A[i, i+1:] / G[i, i]

        # Por simetría, llenamos también la parte triangular inferior de G
        G[i+1:, i] = G[i, i+1:]

        # Paso 3: Actualizar el bloque restante A_JJ <- A_JJ - G_Ji^T * G_iJ
        # G[i+1:, i] es un vector columna y G[i, i+1:] es un vector fila.
        # np.outer calcula el producto externo para obtener la matriz de actualización.
        G_iJ = G[i, i+1:]
        A[i+1:, i+1:] -= np.outer(G_iJ, G_iJ)

    # La guía asume la forma de los bloques donde el resultado es G triangular inferior
    # Nos aseguramos de limpiar la parte superior derecha para devolver G estrictamente triangular inferior
    return np.tril(G)
"""
np.tril(G) es una función de NumPy que toma una matriz y deja intacta la mitad inferior (incluyendo la diagonal),
mientras convierte todo lo que está arriba en ceros.
"""

#CÓDIGO DIRECTO DE PRUEBA
def es_definida_positiva(A):
    # Calcula los autovalores de la matriz
    autovalores = np.linalg.eigvals(A)
    # Verifica si TODOS los autovalores son mayores que cero
    return np.all(autovalores > 0)

# 1. Definimos los bloques básicos de 3x3
B = np.array([
    [ 4, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  4]
])

I = np.eye(3)
O = np.zeros((3, 3))

# 2. Construimos la matriz gigante B* (9x9) uniendo los bloques
B_star = np.block([
    [ B, -I,  O],
    [-I,  B, -I],
    [ O, -I,  B]
])

print("Matriz B* original (9x9):")
print(B_star)
print("-" * 50)

# 3. Validación y Cálculo
if es_definida_positiva(B_star):
    G_star = cholesky(B_star)

    print("Factor de Cholesky G (Matriz Triangular Inferior):")
    print(np.round(G_star, 3))
    print("-" * 50)



