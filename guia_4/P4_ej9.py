import numpy as np
from scipy.linalg import cholesky, solve_triangular, qr

def ejercicio_9(n):
    # 1. Construcción de la matriz A (n x n-2) y vector b (n x 1)
    A = np.zeros((n, n - 2))
    for j in range(n - 2):
        A[j, j] = -1
        A[j + 1, j] = 2
        A[j + 2, j] = -1

    b = np.zeros(n)
    b[0] = 1.0
    b[-1] = 1.0

    # --- Método 1: Cholesky (Ecuaciones Normales) ---
    ATA = A.T @ A
    ATb = A.T @ b
    
    # Factorización ATA = L @ L.T
    L = cholesky(ATA, lower=True)
    y = solve_triangular(L, ATb, lower=True)
    x_cholesky = solve_triangular(L.T, y, lower=False)

    # --- Método 2: Descomposición QR ---
    Q, R = qr(A, mode='economic')
    QTb = Q.T @ b
    x_qr = solve_triangular(R, QTb, lower=False)

    # Comparación de la diferencia entre ambas soluciones
    diferencia = np.linalg.norm(x_cholesky - x_qr)

    return x_cholesky, x_qr, diferencia

# Ejecución para n = 100 y n = 1000
for n in [100, 1000]:
    x_chol, x_qr, diff = ejercicio_9(n)
    print(f"--- Resultados para n = {n} ---")
    print(f"Norma de la diferencia ||x_chol - x_qr||: {diff:.2e}")
    print(f"Primeros 3 elementos de x (QR): {x_qr[:3]}")
    print(f"Últimos 3 elementos de x (QR): {x_qr[-3:]}\n")