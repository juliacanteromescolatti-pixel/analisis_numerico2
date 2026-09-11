import numpy as np

def householder(x):
    m = len(x)
    u = x.copy()
    sigma = sum(x[1:]**2)

    if sigma == 0:
        u = np.zeros(m)
        rho = 0.0
        return u, rho

    mu = np.sqrt(sigma + x[0]**2)

    if x[0] <= 0:
        gamma = x[0] - mu
    else:
        gamma = -sigma / (x[0] + mu)

    rho = (2 * (gamma**2)) / (sigma + (gamma**2))
    u = np.array([1.0] + [x[i] / gamma for i in range(1, m)])

    return u, rho

def qrholder(A):
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)
    
    # CORREGIDO 2: El límite superior debe ser m-1 para evitar procesar un escalar al final
    p = min(m - 1, n)
    
    for j in range(p):
        u, rho = householder(R[j:, j])
        w = rho * u
        
        # CORREGIDO 3: Quitamos el .T porque 'u' es unidimensional
        v = u @ R[j:, j:]
        v1 = Q[:, j:] @ w
        
        R[j:, j:] = R[j:, j:] - np.outer(w, v)
        Q[:, j:] = Q[:, j:] - np.outer(v1, u)

    return Q, R

# ====================================================================================
# --- MATRIZ DE PRUEBA E INCISO B ---
# ====================================================================================
print("==========================================================================================================")
print("VERIFICACIÓN INCISO B - HOUSEHOLDER")
print()

# Matriz clásica de ejemplo (Da coeficientes exactos sin tantos decimales feos)
A_prueba = np.array([
    [12, -51,   4],
    [ 6, 167, -68],
    [-4,  24, -41]
], dtype=float)

Q, R = qrholder(A_prueba)

print("La matriz original A es: ")
print(A_prueba)
print()
print("La matriz Q (Ortogonal) es:")
print(np.round(Q, 4))
print()
print("La matriz R (Triangular Superior) es:")
print(np.round(R, 4))
print()
print("Verificación Q @ R (Debería ser igual a A):")
print(np.round(Q @ R, 4))
print()
print("Diferencia absoluta (A - Q@R):")
print(np.round(A_prueba - Q @ R, 10))
