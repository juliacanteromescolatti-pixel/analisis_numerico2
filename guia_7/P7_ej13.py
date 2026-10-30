import numpy as np
import time
from scipy.sparse import csr_matrix, tril, triu, diags
from scipy.sparse.linalg import spsolve

def gradcon_pcond(A, b, x0, tol, m, M):
    """
    Método de gradiente con precondicionamiento.
    Entradas:
        A   : Matriz del sistema (n x n)
        b   : Vector de términos independientes (n,)
        x0  : Vector inicial (n,)
        tol : Tolerancia (ε)
        m   : Número máximo de iteraciones
        M   : Matriz de precondicionamiento
    Salida:
        x*  : Solución aproximada
    """
    x = np.array(x0, dtype=float)
    b = np.array(b, dtype=float)

    # Paso 0: Definir r = b - Ax^0 y sigma = ||r||_2
    r = b - A @ x
    sigma = np.linalg.norm(r, ord=2)

    # Paso 1: Bucle para k = 1, ..., m
    for k in range(1, m + 1):
        if sigma <= tol:
            break

        # Resolver M * p = r (equivalente a p = M^(-1) * r)
        # Si M es una matriz de scipy.sparse o numpy, spsolve / solve maneja sistemas triangulares/diagonales eficientemente.
        if hasattr(M, "tocsr") or hasattr(M, "toarray"):
            p = spsolve(M, r)
        else:
            p = np.linalg.solve(M, r)

        q = A @ p

        # Cálculo de alpha = (p^T * r) / (p^T * q)
        num = p.T @ r
        den = p.T @ q

        if den == 0:
            raise ValueError("División por cero en el cálculo de alpha (p^T * q = 0).")

        alpha = num / den

        # Actualizaciones
        x = x + alpha * p
        r = r - alpha * q
        sigma = np.linalg.norm(r, ord=2)

    return x