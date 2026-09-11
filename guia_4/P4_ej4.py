import numpy as np
from P4_ej3A import rotacion_givens_bloques

def qrgivensp(A):
    """
    Factorización QR con pivoteo de columnas mediante rotaciones de Givens.
    
    Calcula Q, R, P tales que:
        A @ P = Q @ R
        
    Parámetros:
    -----------
    A : np.ndarray
        Matriz de tamaño (m, n).
        
    Retorna:
    --------
    Q : np.ndarray de tamaño (m, m), ortogonal.
    R : np.ndarray de tamaño (m, n), triangular superior.
    P : np.ndarray de tamaño (n, n), matriz de permutación.
    """
    # Copias de trabajo en punto flotante
    R = np.array(A, dtype=np.float64, copy=True)
    m, n = R.shape
    
    # Inicializar Q como la identidad de m x m
    Q = np.eye(m, dtype=np.float64)
    
    # Inicializar el vector de permutación de columnas: [0, 1, ..., n-1]
    p = np.arange(n)
    
    # Cuadrado de las normas euclídeas de las columnas de R
    col_norms_sq = np.sum(R**2, axis=0)
    #Python va a mirar la primera columna, va a operar hacia abajo, luego pasará a la segunda columna hacia abajo, y así sucesivamente.
    # Número de pasos de reducción (min(m-1, n))
    limite = min(m - 1, n)
    
    for k in range(limite):
        # 1. Pivoteo: buscar la columna j (j >= k) con mayor norma en el subvector R[k:m, j]
        idx_max = k + np.argmax(col_norms_sq[k:])
        
        # Si la mayor columna es no nula y distinta de la posición k, permutamos
        if idx_max != k:
            # Permutar columnas en R
            R[:, [k, idx_max]] = R[:, [idx_max, k]]
            
            # Registrar el intercambio en el vector de índices de permutación
            p[k], p[idx_max] = p[idx_max], p[k]
            
            # Permutar las normas correspondientes
            col_norms_sq[k], col_norms_sq[idx_max] = col_norms_sq[idx_max], col_norms_sq[k]
            
        # Si la norma de la columna seleccionada es prácticamente 0, el resto es 0
        if col_norms_sq[k] < 1e-15:
            continue
            
        # 2. Anular los elementos por debajo de la diagonal en la columna k usando Givens
        for i in range(k + 1, m):
            if R[i, k] != 0.0:
                # Determinar c y s para anular R[i, k] usando el pivote R[k, k]
                c, s = rotacion_givens_bloques(R[k, k], R[i, k])
                
                # Actualizar filas k e i de la matriz R:
                # [R[k, :]] = [ c s] [R[k, :]]
                # [R[i, :]] [-s c] [R[i, :]]
                filas_k_i = np.array([[c, s], [-s, c]]) @ R[[k, i], :]
                R[[k, i], :] = filas_k_i
                
                # Para acumular Q, multiplicamos por la derecha por G^T:
                # Q = Q @ G^T, lo que equivale a rotar las columnas k e i de Q
                cols_k_i = Q[:, [k, i]] @ np.array([[c, -s], [s, c]])
                Q[:, [k, i]] = cols_k_i

        # 3. Actualizar normas euclídeas para el siguiente paso
        if k + 1 < n:
            col_norms_sq[k + 1:] = np.sum(R[k + 1:m, k + 1:]**2, axis=0)

    # Construir la matriz de permutación P (n x n)
    P = np.eye(n)[:, p]
    
    return Q, R, P


# ==========================================
# Ejemplo de uso y verificación
# ==========================================
if __name__ == "__main__":
    A = np.array([
        [1.0, 2.0, 4.0],
        [3.0, 8.0, 14.0],
        [2.0, 6.0, 13.0]
    ])
    
    Q, R, P = qrgivensp(A)
    
    print("Matriz A original:\n", A)
    print("\nMatriz Q (ortogonal):\n", np.round(Q, 4))
    print("\nMatriz R (triangular superior):\n", np.round(R, 4))
    print("\nMatriz P (permutación):\n", P)
    
    # Comprobación de la factorización A @ P == Q @ R
    AP = A @ P
    QR = Q @ R
    print("\n¿A @ P == Q @ R?:", np.allclose(AP, QR))
    
    # Comprobación de ortogonalidad de Q
    print("¿Q.T @ Q == I?:", np.allclose(Q.T @ Q, np.eye(A.shape[0])))