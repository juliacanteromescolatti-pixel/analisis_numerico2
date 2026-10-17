#USAMOS EL EJRICICO 3
import numpy as np
from P6_ej3_profe import autjacobi

#UASEMOS EL EJRCICIO DEL PRCATICO 4, EJERICIO 3A
import numpy as np

def rotacion_givens(a, b):
    """Calcula los valores c y s para anular el elemento b."""
    if b == 0:
        return 1.0, 0.0
    elif abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau**2)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1.0 + tau**2)
        s = c * tau
    return c, s

def qrgivensp(A_in):
    """
    Descomposición QR con permutación de columnas por Rotaciones de Givens.
    Cumple con AP = QR
    """
    # Copiar la matriz de entrada para no modificarla directamente
    A = A_in.copy().astype(float)
    m, n = A.shape

    # Inicializar las matrices de salida
    Q = np.eye(m)
    P = np.eye(n) # Matriz de permutacion

    p = min(m - 1, n)

    for j in range(p):
        # 1. Pivoteo de columnas: Calcular normas desde la fila j hasta m
        # (En Python los índices van de j a m, que se escribe j:  )
        normas_cuadradas = np.sum(A[j:, j:]**2, axis=0)

        # Encontrar la columna con la norma máxima relativa a la posición actual
        k_max_rel = np.argmax(normas_cuadradas)
        k_max = j + k_max_rel  # Convertir a índice global de la matriz

        # Si la columna máxima no es la actual, se intercambian
        if k_max != j:
            # Intercambiar columnas en A (que se convertirá en R)
            A[:, [j, k_max]] = A[:, [k_max, j]]
            # Registrar el intercambio en la matriz de permutación P
            P[:, [j, k_max]] = P[:, [k_max, j]]

        # 2. Bucle original de eliminación por Givens
        for i in range(j + 1, m):
            if A[i, j] != 0:
                # Índices correspondientes a las filas I = {j, i}
                # e índices correspondientes a las columnas J = {j, ..., n-1}
                c, s = rotacion_givens(A[j, j], A[i, j])

                # Construir la matriz de rotación local 2x2
                G = np.array([[c, -s],[s,  c]])

                # Aplicar la rotación a las filas elegidas de A
                A[[j, i], j:] = G @ A[[j, i], j:]

                # Actualizar la matriz Q (multiplicación por la derecha de Q_*,I por G^T)
                Q[:, [j, i]] = Q[:, [j, i]] @ G.T

    # Condición especial de limpieza del final del algoritmo original
    if m <= n and A[m-1, m-1] == 0:
        A[m-1, (m-1):] = -A[m-1, (m-1):]
        Q[:, m-1] = -Q[:, m-1]

    R = A
    return Q, R, P

#==================================================================================================================
import numpy as np
#EJERICIO 4 GUIA 6
# quiero obtener la descomp svd en valores singulares

def dvsingulares(A):
    m, n = A.shape
    Q, R, P = qrgivensp(A) # descomp qr de A
    A_qr = Q@R

    R_mat = R.T @ R # hacemos esto para que la matriz R sea simetrica y poder aplicar la act 3

# aplicamos la act 3, W seria el Q de nuestro codigo
    autovalores, W = autjacobi(R_mat)

# armamos las matrices de la descomp SVD, U, sigma, V
    sigma = np.sqrt(np.maximum(autovalores, 0)) # Evita errores numéricos con ceros negativos
    sigma = np.diag(sigma)

#  Calcular V utilizando la permutación P y los autovectores W
    V = P @ W
#  Calcular la matriz U
# U se obtiene a partir de Q, R, W y los valores singulares inversos
    U = np.zeros((m, m))

# Para evitar división por cero, filtramos los valores singulares significativos
    tol = 1e-12
    r_rang = np.sum(sigma > tol)

    # Calculamos la parte principal de U
    # U_i = Q * (R * W_i) / sigma_i
    if r_rang > 0:
        # Tomamos las submatrices correspondientes
        Sigma_pinv = np.zeros((n, r_rang))
        np.fill_diagonal(Sigma_pinv, 1.0 / sigma[:r_rang])

        # U = Q * R * W_sub * Sigma_pinv
        U[:, :r_rang] = (Q @ (R @ W[:, :r_rang])) @ Sigma_pinv

    # Si m > r_rang, completamos las columnas restantes de U para que sea una matriz ortogonal
    if m > r_rang:
        # Usamos una base ortonormal para el espacio nulo restante (completar con QR o SVD rápida)
        Q_extra, _ = np.linalg.qr(U[:, :r_rang])
        # O de manera más directa, dejamos que un completado ortogonal asigne las columnas faltantes:
        # En la práctica, completar la base se puede hacer con QR sobre una matriz aleatoria proyectada ortogonalmente.
        # Una forma sencilla en NumPy si m == n o m >= n es usar un complemento ortogonal:
        from scipy.linalg import null_space # Opcional si quieres completarlo formalmente
        # O simplemente completando con las columnas que sobren de Q si el espacio lo permite:
        if Q.shape[1] >= m:
            U[:, r_rang:] = Q[:, r_rang:m]
        else:
            # Si Q es de dimensiones menores, completamos con un QR auxiliar
            matriz_aleatoria = np.random.randn(m, m - r_rang)
            # Proyectamos fuera del espacio de U[:, :r_rang]
            matriz_aleatoria -= U[:, :r_rang] @ (U[:, :r_rang].T @ matriz_aleatoria)
            Q_c, _ = np.qr(matriz_aleatoria)
            U[:, r_rang:] = Q_c[:, :m - r_rang]

    return U, sigma, V


# Matriz de prueba A (3x2)
A = np.array([[1.0, 2.0],[3.0, 4.0],[5.0, 6.0]])

# Ejecutas tu función
U, sigma, V = dvsingulares(A)

print("Matriz U")
print(U)
print()
print("Matriz Sigma")
print(sigma)
print()
print("Matriz V")
print(V)

