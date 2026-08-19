#Nos pide relaizar el algoritmo de Cholesky
import numpy as np
def cholesky(A):
    # Convertimos la entrada a un arreglo de NumPy con tipo flotante (float)
    # para evitar errores de división entera si se ingresan números enteros.
    A = np.array(A, dtype=float)
    
    # Obtenemos el tamaño 'n' de la matriz (como es cuadrada, basta con el número de filas).
    n = A.shape[0]
    
    # Creamos una matriz de ceros del mismo tamaño que A. 
    # Aquí iremos rellenando los valores de G fila por fila.
    G = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            # Calculamos la sumatoria de los productos de los elementos ya calculados en la fila i y la fila j:
            # \sum_{k=0}^{j-1} G_{ik} * G_{jk}
            suma = np.sum(G[i, :j] * G[j, :j])

#El range (i+1), recorre las columnas desde la 0 hasta la posición i. 
#Al llegar solo hasta i, nos aseguramos de calcular únicamente la parte triangular inferior, dejando el resto automáticamente en cero.
#G[i, :j] toma los elementos ya calculados en la fila (i) desde la columna (0) hasta (j-1).
#G[j, :j] toma los elementos de la fila (j) desde la columna (0) hasta (j-1).
# Multiplicarlos con * hace el producto elemento a elemento, y np.sum() los suma todos.

            if i == j:
                # Si estamos en la diagonal principal (i == j):
                def_pos_check = A[i, i] - suma
                
                # Control de seguridad: Si el valor dentro de la raíz es <= 0, 
                # significa que la matriz original NO es definida positiva.
                if def_pos_check <= 0:
                    raise ValueError("La matriz no es definida positiva.")
                
                # Si es válido, la entrada de la diagonal es la raíz cuadrada del remanente.
                G[i, j] = np.sqrt(def_pos_check)
            else:
                # Si estamos debajo de la diagonal (i > j):
                # Despejamos G[i, j] dividiendo por el elemento de la diagonal G[j, j].
                G[i, j] = (A[i, j] - suma) / G[j, j]
                
    return G

# Definimos la matriz B de 3x3 que nos da el ejercicio
B = np.array([
    [ 4, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  4]
])

# Creamos una matriz Identidad de 3x3 usando np.eye
I = np.eye(3)

# Creamos una matriz de Ceros de 3x3 usando np.zeros
O = np.zeros((3, 3))

# Recordando que el enunciado define B* como:
# [ B  -I   0]
# [-I   B  -I]
# [ 0  -I   B]

# Construimos la primera fila uniendo horizontalmente B, -I y O.
# El resultado es una fila de dimensiones 3x9.
Fila1 = np.block([ B, -I,  O])

# Construimos la segunda fila uniendo horizontalmente -I, B y -I.
Fila2 = np.block([-I,  B, -I])

# Construimos la tercera fila uniendo horizontalmente O, -I y B.
Fila3 = np.block([ O, -I,  B])

# Finalmente, unimos verticalmente las tres filas para obtener la matriz B* de 9x9.
B_original = np.block([[Fila1], [Fila2], [Fila3]])

# Enviamos nuestra matriz de 9x9 a la función que escribimos antes
G_calculada = cholesky(B_original)

# Multiplicamos G_calculada por su transpuesta (G_calculada.T) 
# para intentar reconstruir la matriz original.
B_reconstruida = np.dot(G_calculada, G_calculada.T)

# Calculamos la diferencia absoluta elemento a elemento entre la original y la reconstruida,
# y nos quedamos con el valor máximo para ver la calidad de la aproximación.
diferencia_max = np.max(np.abs(B_original - B_reconstruida))



print("--- TESTEO EJERCICIO 5 ---")
# Comprueba si G_calculada es igual a su parte triangular inferior (debe dar True)
print(f"¿G es estrictamente triangular inferior?: {np.allclose(G_calculada, np.tril(G_calculada))}")
