# Hacer un codigo de Jacobi para hallar autovalores de una matriz simetrica
# con err = e-10 y mit = 500
import numpy as np
import matplotlib.pyplot as plt

def jacobi(A, x, b):
    z = len(b)
    u = np.zeros(z)
    mit_j = 500
    tol_j = 10**(-10)
    for k in range(mit_j):
        for i in range(z):
            n = 0
            h = 0
            for j in range(i):
                n += A[i, j] * x[j]
            for j in range(i + 1, z):
                h += A[i, j] * x[j]
            u[i] = (b[i] - n - h) / A[i, i]

        if np.sqrt(np.dot(u - x, u - x)) < tol_j:
            return u
        x = u.copy()

    return x

def autjacobi(A, mu=0.0):
    """
    Encuentra el autovalor de A más cercano a 'mu' usando
    el algoritmo Jacobi original para resolver cada paso.
    """
    mit = 500
    tol = 10**(-10)
    if not np.array_equal(A, A.T):
        print("La matriz no es simétrica.")
        return None

    n = A.shape[0]

    # Matriz desplazada: (A - mu*I)
    A_shift = A - mu * np.eye(n)

    # Vector inicial
    x = np.ones(n)
    x = x / np.linalg.norm(x)

    for k in range(mit):
        # Usamos el algoritmo anterior:
        # Resolvemos A_shift * y = x pasando tu función
        y = jacobi(A_shift, x.copy(), x)

        # Normalizamos el resultado
        x_new = y / np.linalg.norm(y)

        # Criterio de parada
        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(x_new + x) < tol:
            x = x_new
            break
        x = x_new

    # Calculamos el autovalor (Cociente de Rayleigh: lambda = x^T * A * x)
    autovalor = np.dot(x, np.dot(A, x))

    return autovalor, x, k + 1

#===========================================================================
# MATRIZ PRUEBA
#===========================================================================

# Matriz simétrica dominante por diagonal
A_simetrica = np.array([
    [10.0,  1.0,  2.0],
    [ 1.0, 12.0, -2.0],
    [ 2.0, -2.0,  8.0]])

b = np.array([6.0, 25.0, -10.0])
x0 = [0.0, 0.0, 0.0]
print("MATRIZ SIMETRICA")
# Prueba con el resolvedor de Jacobi:
sol = jacobi(A_simetrica, x0, b)
print("Solución del sistema:", sol)
print()

# Prueba con el algoritmo de autovalores:
autovalor, autovector, iters_auto = autjacobi(A_simetrica, mu=0.0)
print("Autovalor aproximado matriz simetrica:", autovalor)
print()
print("MATRIZ NO SIMETRICA")

# Matriz no simétrica
A_no_simetrica = np.array([
    [10.0,  5.0,  2.0],
    [ 1.0, 12.0, -2.0],
    [ 2.0, -2.0,  8.0]
])

# Prueba con el algoritmo (imprimirá el aviso de que no es simétrica):
resultado = autjacobi(A_no_simetrica, mu=0.0)
print()

print(resultado)
#