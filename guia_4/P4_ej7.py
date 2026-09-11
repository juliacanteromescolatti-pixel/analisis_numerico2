import numpy as np
from P4_ej4 import qrgivensp  # tu función del ejercicio 4: A @ P = Q @ R


# =============================================================================
# EJERCICIO 7 — CUADRADOS MÍNIMOS VÍA QR
# =============================================================================
#
# EL PROBLEMA (en criollo)
# Tenés un sistema A x = b que en general NO tiene solución exacta.
# Esto pasa típicamente cuando A tiene más filas que columnas (m > n):
# más ecuaciones que incógnitas. Ejemplo clásico: ajustar una recta
# (2 incógnitas: pendiente y ordenada) a 100 puntos (100 ecuaciones).
# No existe ninguna recta que pase EXACTO por los 100 puntos.
#
# Como no hay solución exacta, en vez de "resolver" A x = b buscamos el
# x que hace que el error sea lo más chico posible, midiendo el error
# con la norma euclídea al cuadrado:
#
#       x_barra = argmin_x  || A x - b ||^2
#
# Eso es "cuadrados mínimos": minimizar la suma de los errores al
# cuadrado en cada ecuación.
#
#
# ¿POR QUÉ NO RESOLVER DIRECTO A^T A x = A^T b (ecuación normal)?
# El ejercicio 6 demostró que el x que minimiza ||Ax-b||^2 es
# exactamente la solución de la "ecuación normal":
#
#       A^T A x = A^T b
#
# Esto es una tentación: armar la matriz cuadrada A^T A (n x n) y
# resolver ese sistema con Cholesky o LU, como hiciste en el
# ejercicio 8. El problema es puramente NUMÉRICO (no matemático): se
# puede demostrar que
#
#       cond(A^T A) = cond(A)^2
#
# Es decir, si A ya estaba algo mal condicionada, A^T A está
# MUCHÍSIMO peor condicionada (el cuadrado de un número ya grande).
# Cuanto peor condicionado está un sistema, más precisión perdés al
# resolverlo en punto flotante. Por eso, en vez de pasar por A^T A,
# usamos la factorización QR de A directamente: el error que se
# comete al resolver via QR crece proporcional a cond(A), NO a
# cond(A)^2. Es el mismo resultado matemático, pero llegás con mucha
# más precisión. (Esto es justo lo que se ve comparando los métodos
# en el ejercicio 8.)
#
#
# LA IDEA DE USAR QR: DE UN PROBLEMA RECTANGULAR A UNO TRIANGULAR
# Factorizamos A con pivoteo de columnas (ejercicio 4):
#
#       A P = Q R          (Q ortogonal, R triangular superior, P permutación)
#
# PASO CLAVE (el "truco" de todo el ejercicio): multiplicar un vector
# por una matriz ORTOGONAL no cambia su norma. Geométricamente, Q
# representa una rotación/reflexión, y rotar o reflejar un vector no
# lo estira ni lo encoge. Formalmente:
#
#       || Q^T v ||^2 = v^T Q Q^T v = v^T v = || v ||^2      (porque Q^T Q = I)
#
# Entonces puedo "meter" un Q^T adentro de la norma sin cambiar nada:
#
#       || A x - b ||  =  || Q^T (A x - b) ||
#
# Ahora reemplazo A por Q R P^T (despejando de A P = Q R, usando que
# P es ortogonal así que P^{-1} = P^T):
#
#       Q^T (A x - b) = Q^T (Q R P^T) x - Q^T b
#                     = (Q^T Q) R P^T x - Q^T b
#                     = R (P^T x) - Q^T b                    (porque Q^T Q = I)
#
# Si definimos el CAMBIO DE VARIABLE  y = P^T x  (que es simplemente x
# con sus componentes reordenadas según la permutación P), el problema
# original se transformó en:
#
#       min_y  || R y - c ||^2          con   c := Q^T b
#
# y esto es mucho más fácil, porque R es triangular en vez de una
# matriz cualquiera.
#
#
# ¿POR QUÉ ALCANZA CON RESOLVER UN SISTEMA TRIANGULAR CHICO?
# Supongamos A de tamaño (m, n) con m >= n (más filas que columnas,
# el caso típico de cuadrados mínimos) y rango columna completo.
# Entonces R tiene esta pinta, partida en bloques:
#
#           [ R1 ]  <- (n, n)   triangular superior "de verdad"
#       R = [    ]
#           [ 0  ]  <- (m-n, n) debería ser ~0 (por construcción de la QR)
#
# y c = Q^T b también se parte en dos:
#
#           [ c1 ]  <- (n,)     primeras n componentes
#       c = [    ]
#           [ c2 ]  <- (m-n,)   resto
#
# Como la parte de abajo de R es (idealmente) todo ceros, al calcular
# R y para cualquier y, las primeras n filas dan R1 y, y las últimas
# m-n filas dan 0 (siempre, sin importar y). Por lo tanto:
#
#       || R y - c ||^2  =  || R1 y - c1 ||^2  +  || 0 - c2 ||^2
#                        =  || R1 y - c1 ||^2  +  || c2 ||^2
#
# ¡Esto es la clave! El segundo término, ||c2||^2, NO DEPENDE DE y.
# Es un número fijo: es el error mínimo que el sistema va a tener sí o
# sí (recordá que el sistema original no tenía solución exacta, así
# que algo de error siempre va a quedar).
#
# El PRIMER término, en cambio, sí depende de y, y lo podemos hacer
# EXACTAMENTE CERO, porque R1 es una matriz cuadrada (n x n) e
# inversible (si A tiene rango completo). Entonces basta con resolver
# el sistema cuadrado:
#
#       R1 y = c1
#
# y ya minimizamos todo lo que se podía minimizar. Como R1 es
# triangular superior, resolver este sistema es barato: sustitución
# hacia atrás (ver la función de abajo).
#
# Por último, deshacemos el cambio de variable: como y = P^T x,
# despejando (P es ortogonal, P^{-1} = P^T):
#
#       x = P y
#
# Y ESO es lo que devuelve sol_cuadmin.
# =============================================================================


def resolver_triangular_superior(R1, c1):
    """
    Resuelve el sistema triangular superior  R1 @ y = c1  mediante
    SUSTITUCIÓN HACIA ATRÁS (back-substitution).

    R1 : np.ndarray, forma (n, n), triangular superior
         (lo que está debajo de la diagonal se ignora / es ~0).
    c1 : np.ndarray, forma (n,)

    IDEA DEL ALGORITMO, con un ejemplo concreto n=3:
    Un sistema triangular superior se ve así:

        R1[0,0]*y0 + R1[0,1]*y1 + R1[0,2]*y2 = c1[0]     <- fila 0
                     R1[1,1]*y1 + R1[1,2]*y2 = c1[1]     <- fila 1
                                  R1[2,2]*y2 = c1[2]     <- fila 2

    Fijate que la ÚLTIMA fila (fila 2) tiene UNA SOLA incógnita: y2.
    Entonces la despejamos directo:

        y2 = c1[2] / R1[2,2]

    Ahora que ya conocemos y2 (es un número, no una incógnita), la
    PENÚLTIMA fila (fila 1) también queda con una sola incógnita
    desconocida (y1), porque y2 ya lo podemos reemplazar:

        y1 = ( c1[1] - R1[1,2]*y2 ) / R1[1,1]

    Y así seguimos subiendo, fila por fila, hasta llegar a la fila 0,
    donde y1 e y2 ya son conocidos:

        y0 = ( c1[0] - R1[0,1]*y1 - R1[0,2]*y2 ) / R1[0,0]

    Por eso el algoritmo se llama "hacia atrás": vas resolviendo desde
    la ÚLTIMA fila hacia la PRIMERA (al revés de cómo se suele escribir
    el sistema). En código, por eso el for recorre
    range(n-1, -1, -1) = [n-1, n-2, ..., 1, 0].
    """
    n = R1.shape[0]
    y = np.zeros(n)  # acá vamos a ir guardando las soluciones que calculamos

    # i recorre n-1, n-2, ..., 1, 0  (de la última fila a la primera)
    for i in range(n - 1, -1, -1):

        # R1[i, i+1:]  son los coeficientes de la fila i que multiplican
        # a las incógnitas y[i+1], y[i+2], ..., y[n-1] -- TODAS ellas ya
        # las calculamos en vueltas anteriores del for (porque i+1 > i
        # significa "filas ya procesadas").
        #
        # y[i+1:]  son justamente esos valores ya calculados.
        #
        # El producto punto R1[i, i+1:] @ y[i+1:]  calcula exactamente
        # la suma  R1[i,i+1]*y[i+1] + R1[i,i+2]*y[i+2] + ... + R1[i,n-1]*y[n-1]
        # osea, "todo lo que ya sabemos" de la ecuación i.
        suma_terminos_conocidos = R1[i, i + 1:] @ y[i + 1:]

        # La ecuación i completa es:
        #     R1[i,i]*y[i] + suma_terminos_conocidos = c1[i]
        # Despejamos y[i]:
        y[i] = (c1[i] - suma_terminos_conocidos) / R1[i, i]

    return y


def sol_cuadmin(A, b):
    """
    Resuelve el problema de cuadrados mínimos:

            x_barra = argmin_x  || A x - b ||_2^2

    usando la factorización QR con pivoteo de columnas del ejercicio 4
    (A P = Q R) y un sistema triangular (ver explicación completa
    arriba, antes de resolver_triangular_superior).

    Parámetros
    A : np.ndarray, forma (m, n)   -- se asume m >= n y rango columna completo
    b : np.ndarray, forma (m,)

    Retorna
    x : np.ndarray, forma (n,) -- el x_barra que minimiza || A x - b ||^2
    """
    # Convertimos a float64 por si A o b vinieran como enteros
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    m, n = A.shape

    #PASO 1: factorizar  A P = Q R  (del ejercicio 4)
    Q, R, P = qrgivensp(A)

    #PASO 2: pasar b al "sistema de referencia" de Q
    # c = Q^T b. Esto es lo que en la explicación de arriba llamamos
    # "meter Q^T adentro de la norma sin cambiar nada".
    c = Q.T @ b

    #PASO 3: separar en bloques (R1, c1) vs (0, c2)
    # Solo nos interesan las primeras n filas: ahí vive R1 (n x n,
    # triangular superior de verdad) y c1 (las primeras n componentes
    # de c). El resto de R (filas n en adelante) es ~0, y el resto de
    # c (c2) es el residuo mínimo que no podemos evitar -- no lo
    # necesitamos para calcular x, solo si quisiéramos reportar el
    # error mínimo ||c2||^2 (ver ejercicio 5).
    R1 = R[:n, :n]
    c1 = c[:n]

    #PASO 4: resolver el sistema triangular  R1 y = c1
    y = resolver_triangular_superior(R1, c1)

    #PASO 5: deshacer el cambio de variable  x = P y
    # Recordá: y = P^T x  =>  x = P y  (porque P ortogonal, P^{-1}=P^T)
    x = P @ y

    return x


# =============================================================================
# EJEMPLO PASO A PASO (mismo que se muestra en el chat, para que lo puedas
# correr vos y ver cada número)
# =============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Sistema con 3 ecuaciones y 2 incógnitas: no tiene solución exacta.
    A = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
    b = np.array([1.0, 2.0, 4.0])

    print("A =\n", A)
    print("b =", b)
    print()

    Q, R, P = qrgivensp(A)
    print("Q (ortogonal) =\n", Q)
    print("R (triangular) =\n", R)
    print("P (permutación) =\n", P)
    print()

    c = Q.T @ b
    n = A.shape[1]
    print("c = Q^T b =", c)
    print("  -> c1 (primeras n) =", c[:n], " | c2 (resto, error irreducible) =", c[n:])
    print()

    R1 = R[:n, :n]
    c1 = c[:n]
    print("Sistema triangular a resolver:  R1 @ y = c1")
    print("R1 =\n", R1)
    print("c1 =", c1)
    print()

    x = sol_cuadmin(A, b)
    print("x_barra =", x)

    # Verificación contra numpy
    x_np, *_ = np.linalg.lstsq(A, b, rcond=None)
    print("numpy.linalg.lstsq:", x_np)
    print("¿coinciden?:", np.allclose(x, x_np))

    print()
    print("Error mínimo (irreducible) = ||c2||^2 =", np.sum(c[n:]**2))
    print("Comparar con || A@x_barra - b ||^2   =", np.sum((A @ x - b) ** 2))