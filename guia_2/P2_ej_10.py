import numpy as np
# quiero a los ejercicios del algoritmo 5 y 6, agregarles permutacion,
# y testearlos

# =======================================================================
# 1. GAUSS CON PERMUTACIÓN (Ejercicio 5 + Permutación)
# =======================================================================
"""
ENTRADA: matriz A y vector b
SALIDA: matriz U y vector y de un sistema equivalente
"""

def egaussp(A, b):
    n = len(b)
    A_gauss = A.copy().astype(float)
    # Convertimos 'b' a un vector columna (n, 1) de tipo flotante
    b_gauss = b.copy().astype(float).reshape(-1, 1)

    for k in range(n - 1):
        l = k + np.argmax(np.abs(A_gauss[k:, k]))

        if l != k:
            # Intercambio las filas l y k y despues actualizo la matriz
            A_gauss[[l, k], :] = A_gauss[[k, l], :] #Los : son para indicar que son las filas
            b_gauss[[l, k]] = b_gauss[[k, l]] #No lleva -. pues es un solo vector

        for i in range(k + 1, n):
            vi = A_gauss[i, k] / A_gauss[k, k]
            A_gauss[i, k] = 0.0
            #Modifica toda la fila i a la derecha de la columna k.
            A_gauss[i, k + 1 :] = A_gauss[i, k + 1 :] - vi * A_gauss[k, k + 1 :]
            b_gauss[i] = b_gauss[i] - vi * b_gauss[k]

    U = A_gauss
    y = b_gauss.flatten()
    return U, y

# MATRIZ DE PRUEBA
print("MATRIZ DE PRUEBA PARA GAUSS")
print()
A = np.array([[0., 2, 1], [4., -1, 3], [2.,  8, -2]])
print(A)
print()
b = np.array([5.,11.,6.])
print("El vector b es: ")
print(b)
print()
sol = egaussp(A, b)
print("====================================================================")
print()
print("SOLUCION")
print("La matriz U es: ")
print(sol[0])
print()
print("El vector y es: ")
print(sol[1])
print()
print("====================================================================")
print()


# =======================================================================
# 2. LU CON PERMUTACIÓN (Ejercicio 6 + Permutación)
# =======================================================================
def dlup(A):
    A_lu = A.copy().astype(float)
    n = len(A_lu)
    P = np.eye(n)

    for k in range(n - 1):
        l = k + np.argmax(np.abs(A_lu[k:, k]))

        if l != k:
            A_lu[[l, k], :] = A_lu[[k, l], :]
            P[[l, k], :] = P[[k, l], :]

        for i in range(k + 1, n):
            A_lu[i, k] = A_lu[i, k] / A_lu[k, k]
            A_lu[i, k + 1 :] = A_lu[i, k + 1 :] - A_lu[i, k] * A_lu[k, k + 1 :]

    return A_lu, P

# MATRIZ DE PRUEBA
print("MATRIZ DE PRUEBA PARA GAUSS")
A = np.array([[0.0, 2, 1], [4.0, -1, 3], [2.0, 8, -2]])
print(A)
print("solucion")
sol, P = dlup(A)
print("Matriz A compacta (L y U):", sol)
print("Matriz P:", P)

# Verificación de que P @ A == L @ U
L = np.tril(sol, -1) + np.eye(len(A))
U = np.triu(sol)
print("¿P @ A == L @ U?:", np.allclose(P @ A, L @ U))


"""OBSERVACION: SOBRE EL RECORTE EN LA ACTUALIZACIÓN DE FILAS:
A[i, k + 1 :] = A[i, k + 1 :] - vi * A[k, k + 1 :]
1. Operación Elemental por Filas:
    Matemáticamente, la eliminación gaussiana resta un múltiplo de la fila
    pivote 'k' a TODA la fila 'i' (Fi <- Fi - vi * Fk), no solo a un elemento.

2. ¿Por qué NO usar A[i, i] = ...?
    Escribir A[i, i] solo actualizaría el elemento de la diagonal principal,
    dejando intactos e incorrectos los demás coeficientes de esa fila a la derecha.

3. ¿Por qué usar el recorte [k + 1 :]?
    - Los elementos a la izquierda de 'k' ya son 0 (o guardan multiplicadores de L).
    - El elemento en 'k' se convierte en 0.0 explícitamente.
    - El rango [k + 1 :] selecciona simultáneamente TODAS las columnas restantes
        a la derecha del pivote, permitiendo a NumPy actualizar la fila entera
        en una sola operación vectorial eficiente. """