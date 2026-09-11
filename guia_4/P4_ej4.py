import numpy as np
from P4_ej3A import rotacion_givens_bloques  

def qrgivensp(A):
    """
    Factorización QR CON PIVOTEO DE COLUMNAS usando rotaciones de Givens.

    Idea general del algoritmo:
    ----------------------------
    En la QR "común" (sin pivoteo) vas recorriendo columna por columna,
    de izquierda a derecha, y anulás los elementos debajo de la diagonal.

    Acá, ANTES de trabajar la columna k, elegís entre las columnas que
    quedan (k, k+1, ..., n-1) la que tenga la mayor norma en la parte
    "no procesada todavía" (filas k en adelante), y la ponés en la
    posición k mediante una permutación. Esto es lo que se llama
    "pivoteo por columnas": sirve para mejorar la estabilidad numérica
    y para detectar rango deficiente (si en algún paso todas las normas
    restantes son ~0, significa que las columnas que quedan son
    combinación lineal de las anteriores).

    Al final obtenés:
        A @ P = Q @ R
    en vez de la relación usual A = QR, porque las columnas de A fueron
    reordenadas por P antes de aplicar las rotaciones.

    Parámetros
    ----------
    A : np.ndarray, forma (m, n)
        Matriz a factorizar.

    Retorna
    -------
    Q : np.ndarray, forma (m, m)   -> matriz ortogonal (Q^T Q = I)
    R : np.ndarray, forma (m, n)   -> matriz triangular superior
    P : np.ndarray, forma (n, n)   -> matriz de permutación (0-1, una sola
                                    entrada "1" por fila y por columna)
    """

    # -----------------------------------------------------------------
    # PASO 0: inicialización
    # -----------------------------------------------------------------

    # Trabajamos sobre una COPIA de A (no queremos modificar la matriz
    # original que pasó el usuario) y forzamos float64 por si A viniera
    # con enteros (si no, las cuentas con Givens podrían truncarse).
    R = np.array(A, dtype=np.float64, copy=True)
    m, n = R.shape

    # Q arranca en la identidad. La idea es que en cada rotación de
    # Givens que aplicamos a R por izquierda (para anular una entrada),
    # vamos "deshaciendo" esa misma rotación sobre Q por derecha.
    # Al final: Q = G1^T @ G2^T @ ... (producto de las transpuestas de
    # todas las rotaciones aplicadas), y eso es justamente lo que hace
    # que A = Q @ R (o A@P = Q@R en este caso con pivoteo).
    Q = np.eye(m, dtype=np.float64)

    # p es el "vector de permutación": p[k] guarda, en cada momento,
    # cuál era el índice ORIGINAL de la columna que ahora está en la
    # posición k de R. Arranca como [0, 1, ..., n-1] (sin permutar).
    p = np.arange(n)

    # col_norms_sq[j] = ||R[:, j]||^2 (norma euclídea al cuadrado de la
    # columna j). Se usa al cuadrado para no calcular raíces de más
    # (para comparar cuál es mayor, no hace falta la raíz).
    col_norms_sq = np.sum(R**2, axis=0)

    # Número de columnas que realmente vamos a "pivotear y anular".
    # Igual que en la QR sin pivoteo: como mucho podés anular hasta
    # m-1 filas por columna (la última fila de la "ventana" activa no
    # necesita rotación), y como mucho hay n columnas.
    limite = min(m - 1, n)

    # -----------------------------------------------------------------
    # BUCLE PRINCIPAL: una iteración por cada columna "pivote" k
    # -----------------------------------------------------------------
    for k in range(limite):

        # --- 1. ELEGIR LA COLUMNA PIVOTE ---
        # Entre las columnas k, k+1, ..., n-1, buscamos cuál tiene la
        # mayor norma (al cuadrado) TODAVÍA. col_norms_sq[k:] es el
        # sub-array de esas normas, y argmax me da la posición dentro
        # de ese sub-array; le sumo k para volver a la posición real
        # dentro de col_norms_sq completo.
        idx_max = k + np.argmax(col_norms_sq[k:])

        # Si la columna con mayor norma no es la que ya está en la
        # posición k, hay que intercambiarlas (pivotear):
        if idx_max != k:
            # (a) Intercambiamos físicamente las columnas k e idx_max en R
            R[:, [k, idx_max]] = R[:, [idx_max, k]]

            # (b) Actualizamos el registro de permutación: anotamos que
            #     lo que estaba en la posición k ahora está en idx_max
            #     y viceversa. Esto es lo que después usamos para
            #     reconstruir la matriz P.
            p[k], p[idx_max] = p[idx_max], p[k]

            # (c) Como ya calculamos las normas antes, en vez de
            #     recalcularlas simplemente las intercambiamos también
            #     (más barato que volver a sumar cuadrados).
            col_norms_sq[k], col_norms_sq[idx_max] = col_norms_sq[idx_max], col_norms_sq[k]

        # --- Caso columna (numéricamente) nula ---
        # Si la norma de la columna que quedó en la posición k es
        # prácticamente 0, quiere decir que esa columna (restringida a
        # las filas que faltan procesar) ya es cero: no hay nada que
        # anular con rotaciones. Como además es la de mayor norma entre
        # las que quedan, TODAS las columnas restantes también son ~0
        # en esa parte la matriz es de rango deficiente a partir de
        # acá. "continue" salta directamente a la siguiente k sin hacer
        # rotaciones (R ya tiene ceros ahí).
        if col_norms_sq[k] < 1e-15:
            continue

        # --- 2. ANULAR LOS ELEMENTOS DEBAJO DE LA DIAGONAL EN LA COLUMNA k ---
        # Recorremos cada fila i por debajo de la diagonal (i = k+1 ... m-1)
        # y, de a una por vez, usamos una rotación de Givens para anular
        # la entrada R[i, k] usando como "referencia" la entrada R[k, k].
        for i in range(k + 1, m):
            if R[i, k] != 0.0:  # si ya es 0, no hace falta rotar

                # rotacion_givens_bloques nos da c, s tales que aplicar
                #   [[ c, s],
                #    [-s, c]]
                # al par (R[k,k], R[i,k]) deja el segundo elemento en 0.
                c, s = rotacion_givens_bloques(R[k, k], R[i, k])

                # --- Actualización de R (rotación por IZQUIERDA) ---
                # Solo nos interesan las filas k e i (las demás filas no
                # cambian). Tomamos esas dos filas completas de R,
                # multiplicamos por la matriz de rotación 2x2, y
                # reemplazamos esas dos filas por el resultado:
                #
                #   [ R[k,:] ]      [ c  s] [ R[k,:] ]
                #   [ R[i,:] ]  <-  [-s  c] [ R[i,:] ]
                #
                # Después de esto, R[i, k] queda en 0 (por construcción
                # de c, s), y el resto de las entradas de esas dos filas
                # se actualizan de forma consistente.
                filas_k_i = np.array([[c, s], [-s, c]]) @ R[[k, i], :]
                R[[k, i], :] = filas_k_i

                # --- Actualización de Q (rotación por DERECHA) ---
                # Cada vez que aplicamos G por izquierda a R (para ir
                # armando G_p...G_1 A = R), tenemos que acumular en Q la
                # transpuesta de esa rotación por derecha, para que al
                # final A = Q R.
                # G^T para una rotación de Givens es simplemente la
                # rotación "inversa":
                #   G = [[ c, s], [-s, c]]  =>  G^T = [[c, -s], [s, c]]
                # Acá en vez de tocar filas, tocamos las columnas k e i
                # de Q (porque es multiplicación por derecha):
                #
                #   [Q[:,k]  Q[:,i]]  <-  [Q[:,k]  Q[:,i]] @ [[c, -s], [s, c]]
                cols_k_i = Q[:, [k, i]] @ np.array([[c, -s], [s, c]])
                Q[:, [k, i]] = cols_k_i

        # --- 3. ACTUALIZAR LAS NORMAS PARA LA PRÓXIMA ITERACIÓN ---
        # Como recién modificamos filas de R (las rotaciones cambian
        # TODAS las columnas de las filas k e i, no solo la columna k),
        # las normas de las columnas que faltan procesar (k+1 en
        # adelante) cambiaron. Por eso las recalculamos desde cero,
        # pero solo usando la parte de la matriz que todavía falta
        # procesar: filas k+1 en adelante, columnas k+1 en adelante
        # (la "submatriz activa").
        if k + 1 < n:
            col_norms_sq[k + 1:] = np.sum(R[k + 1:m, k + 1:]**2, axis=0)

    # -----------------------------------------------------------------
    # CONSTRUCCIÓN DE LA MATRIZ DE PERMUTACIÓN P
    # -----------------------------------------------------------------
    # p[k] = índice ORIGINAL de la columna que terminó en la posición k.
    # np.eye(n)[:, p] arma una matriz identidad y reordena SUS COLUMNAS
    # según p. El resultado es la matriz P tal que:
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