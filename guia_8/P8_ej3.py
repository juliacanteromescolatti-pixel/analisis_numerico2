import numpy as np
from scipy.sparse.linalg import gmres

def sol_newton(func, x0, tol=1e-6, m=100):
    """
    Ejecuta el Método de Newton Exacto para resolver F(x) = 0.

    Parámetros:
    -----------
    func : función
        Función que recibe un vector x y retorna una tupla (F, J), donde:
        - F: ndarray de dimensión (n,), evaluación del sistema F(x).
        - J: ndarray de dimensión (n, n), matriz Jacobiana J(x).
    x0 : ndarray
        Punto inicial de la iteración x^0.
    tol : float, opcional
        Tolerancia de error epsilon (criterio de parada sobre ||F(x)||_2).
    m : int, opcional
        Cantidad máxima de iteraciones permitidas.

    Retorna:
    --------
    x : ndarray
        Aproximación de la solución x*.
    k : int
        Número de iteraciones realizadas.
    """
    x = np.array(x0, dtype=float)

    for k in range(m):
        # Evaluación del vector F(x) y la matriz Jacobiana J(x)
        F, J = func(x)

        # Criterio de parada: norma 2 del residuo no lineal
        if np.linalg.norm(F, 2) < tol:
            return x, k

        # Resolución exacta del sistema lineal: J(x^k) * d^k = -F(x^k)
        d = np.linalg.solve(J, -F)

        # Actualización de la posición: x^(k+1) = x^k + d^k
        x = x + d

    print("Advertencia: Se alcanzó la cantidad máxima de iteraciones m.")
    return x, m


def sol_newton_trun(func, x0, tol=1e-6, m=100):
    """
    Ejecuta el Método de Newton Truncado (Inexacto) para resolver F(x) = 0.

    Resuelve el sistema lineal para la dirección de descenso mediante GMRES
    con una tolerancia adaptativa (condición de Eisenstat-Walker).
    """
    x = np.array(x0, dtype=float)

    for k in range(m):
        # Evaluación del vector F(x) y la matriz Jacobiana J(x)
        F, J = func(x)
        norm_F = np.linalg.norm(F, 2)

        # Criterio de parada no lineal
        if norm_F < tol:
            return x, k

        # Cálculo del término de forzado (eta) para el solver lineal
        eta = min(0.5, np.sqrt(norm_F))
        tol_lin = eta * norm_F  # Tolerancia absoluta requerida para GMRES

        # Resolución aproximada (truncada) del sistema lineal J * d = -F
        d, _ = gmres(J, -F, atol=tol_lin)

        # Actualización de la posición
        x = x + d

    print("Advertencia: Se alcanzó la cantidad máxima de iteraciones m.")
    return x, m


    # -------------------------------------------------------------------------
# Definición de la función de prueba (Ejercicio 1)
# Retorna el vector de residuos F(x) y la matriz Jacobiana J(x)
# -------------------------------------------------------------------------
def func_ejercicio1(x):
    # F(x1, x2) = [x1^2 + x2^2 - 4, e^(x1) + x2 - 1]
    F = np.array([
        x[0]**2 + x[1]**2 - 4.0,
        np.exp(x[0]) + x[1] - 1.0
    ])

    # Matriz Jacobiana J(x)
    J = np.array([
        [2.0 * x[0], 2.0 * x[1]],
        [np.exp(x[0]), 1.0]
    ])
    return F, J

# -------------------------------------------------------------------------
# Parámetros de prueba
# -------------------------------------------------------------------------
x0 = np.array([1.0, -1.0])  # Punto inicial
epsilon = 1e-8              # Tolerancia de error
max_iter = 50               # Iteraciones máximas m

# -------------------------------------------------------------------------
# Ejecución y Verificación
# -------------------------------------------------------------------------
sol_exacta, iter_exacta = sol_newton(func_ejercicio1, x0, tol=epsilon, m=max_iter)
sol_trunc, iter_trunc = sol_newton_trun(func_ejercicio1, x0, tol=epsilon, m=max_iter)

print("=== RESULTADOS NEWTON EXACTO ===")
print(f"Solución x*: {sol_exacta}")
print(f"Iteraciones: {iter_exacta}\n")

print("=== RESULTADOS NEWTON TRUNCADO ===")
print(f"Solución x*: {sol_trunc}")
print(f"Iteraciones: {iter_trunc}")