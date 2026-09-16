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
                G = np.array([[c, -s],
                              [s,  c]])

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

#===================================================================================================
#ACLARACION DEL ULTIMO IF
#Porque la matriz \(Q\) tiene que ser una matriz ortogonal perfecta (una matriz con propiedades geométricas muy estrictas,
#como que sus columnas deben medir 1 y ser perpendiculares entre sí).
#Cuando aparece ese 0 al final, el camino matemático normal deja a la matriz \(Q\) "deforme" (con los signos volteados a nivel interno).
# Ese if es como darle un pequeño golpe al rompecabezas para que la pieza \(Q\) recupere su forma geométrica perfecta,
# usando el truco de "menos por menos" para asegurarse de no alterar el resultado del producto.
# ====================================================================================================




# --- Ejemplo de uso y verificación ---
if __name__ == "__main__":
    # Definimos una matriz de ejemplo (con deficiencia de rango o columnas desordenadas)
    A_ejemplo = np.array([
        [1.0, 4.0, 7.0],
        [2.0, 5.0, 8.0],
        [3.0, 6.0, 9.0]
    ])

    Q, R, P = qrgivensp(A_ejemplo)

    print("Matriz R (Triangular superior y ordenada por magnitud en la diagonal):")
    print(np.round(R, 4))

    print("\nVerificación matemática (AP - QR debe ser prácticamente 0):")
    error = (A_ejemplo @ P) - (Q @ R)
    print(np.round(error, 10))
