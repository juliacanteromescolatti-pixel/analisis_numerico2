import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
#---------------------------------------------------------
# EJERCICIO NECESARIO
#---------------------------------------------------------

# EJERCICIO 11 P 7


def arnoldi(A, v, m):
    A = np.array(A, dtype=float)
    n = A.shape[0] # Dimensión de la matriz A (n x n)
    v = np.array(v, dtype=float)

    if m > n:
        raise ValueError("m debe ser menor o igual que n")

    # Inicialización de las matrices de salida según el algoritmo
    V = np.zeros((n, m))
    H_bar = np.zeros((m + 1, m))

    # Paso 0: Si la norma del vector inicial es 0
    if np.linalg.norm(v, ord=2) == 0:
        return H_bar, V

    # Paso 1: Bucle principal para j = 1, ..., m (en Python de 0 a m-1)
    for j in range(m):
        norm_v = np.linalg.norm(v, ord=2)
        if norm_v == 0:
            break

        # Normalización y asignación de la columna j-ésima de V
        V[:, j] = v / norm_v

        # Primer producto con A
        v = A @ V[:, j]

        # 2. Ortogonalización (Bucle interno para i = 1, ..., j)
        for i in range(j + 1):  # En Python los índices van de 0 a j
            H_bar[i, j] = V[:, i].T @ v
            v = v - H_bar[i, j] * V[:, i]

        # 3. Comprobación y cálculo del elemento subdiagonal
        norm_v = np.linalg.norm(v, ord=2)
        if norm_v == 0:
            break
        else:
            H_bar[j + 1, j] = norm_v

    return H_bar, V

#---------------------------------------------------------
# EJERCICIO NECESARIO
#---------------------------------------------------------

# EJERCICIO 7 P 4
# TRIANG SUP POR FILAS
def soltrsup_f(A, b):
	n = len(b)

#Verifico la singularidad:
	if np.any(np.diag(A) == 0):
		print("Error: la matriz es singular")
		return None

	x = np.zeros(n)

	for i in range(n-1, -1, -1): #ponemos -1 -1 pq hace que se recorran las filas desde la ultima a la primera
		suma = 0
		for j in range(i+1, n):
			suma += A[i,j]* x[j]

		x[i] = (b[i]-suma)/ A[i,i]

	return x

# TRIANG SUP POR FILAS POR COLUMNAS
def soltrsup_c(A, b):
    n = len(b)

    # Verifico la singularidad:
    if np.any(np.diag(A) == 0):
        print("Error: la matriz es singular")
        return None

    x = np.zeros(n)
    # Hacemos una copia decimal de b para restarle los efectos de las columnas
    b_actualizado = b.astype(float) # Como restamos terminos similares para actualizar el b entonces lo pasamos a float

    # Recorremos por columnas de izquierda a derecha (0 hasta n-1)
    for j in range(n-1, -1, -1):
        # 1. Despejamos la incógnita actual usando el b modificado
        x[j] = b_actualizado[j] / A[j, j]

        # 2. Restamos el efecto de esta columna en las filas de abajo
        #if j < n - 1:
            # A[j+1:, j] toma toda la columna j por debajo de la diagonal
        b_actualizado[:j] -= A[:j, j] * x[j]
    return x

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
                G = np.array([[c, -s], [s,  c]])

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

def sol_cuadmin(A, b):
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
    y = soltrsup_f(R1, c1)

    #PASO 5: deshacer el cambio de variable  x = P y
    # Recordá: y = P^T x  =>  x = P y  (porque P ortogonal, P^{-1}=P^T)
    x = P @ y

    return x

#========================================================================================
#EJERCICIO SOLICITADO

def sol_gmres(A, b, x, err, mit, mA):
    n = A.shape[0]
    r = b - A @ x
    sigma =np.linalg.norm(r, ord = 2)
    if mA >> n:
        raise ValueError("m debe ser menor o igual que n")

    for k in range(mit):
        if sigma <= err:
            x_mas = x
        return x_mas
    else:
        e1 = np.zeros(mA + 1)
        e1[0] = 1.0
        sigma_e1 = sigma * e1
        H_bar, V = arnoldi(A, r, mA)
        alpha, sigma_mas = sol_cuadmin(H_bar, sigma_e1)
        x = x + V * alpha
        r = b - A @ x
        x_mas = x
        return x_mas

# -.-.-.-.-.-.-.-.-.--.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#  APLICACION  (chequear el resto esta bien)
# -.-.-.-.-.-.-.-.-.--.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

# 1. Función calor(n) basada en el Ejercicio 9
def calor(n):
    N = n - 2
    N2 = N * N

    data = []
    row_ind = []
    col_ind = []
    b = np.zeros(N2)

    for j in range(2, n):
        for i in range(2, n):
            k = (j - 2) * N + (i - 2)

            # Diagonal principal
            data.append(4.0)
            row_ind.append(k)
            col_ind.append(k)

            # Vecino izquierdo
            if i - 1 == 1:
                b[k] += 100.0
            else:
                k_left = (j - 2) * N + (i - 1 - 2)
                data.append(-1.0)
                row_ind.append(k)
                col_ind.append(k_left)

            # Vecino derecho
            if i + 1 < n:
                k_right = (j - 2) * N + (i + 1 - 2)
                data.append(-1.0)
                row_ind.append(k)
                col_ind.append(k_right)

            # Vecino inferior
            if j - 1 > 1:
                k_down = (j - 1 - 2) * N + (i - 2)
                data.append(-1.0)
                row_ind.append(k)
                col_ind.append(k_down)

            # Vecino superior
            if j + 1 < n:
                k_up = (j + 1 - 2) * N + (i - 2)
                data.append(-1.0)
                row_ind.append(k)
                col_ind.append(k_up)

    A = csr_matrix((data, (row_ind, col_ind)), shape=(N2, N2))
    return A, b
# 2. Implementación de los métodos iterativos (Ejercicios 2, 5, etc.)
def sol_jacobi_sparse(A, b, tol=1e-5, mit=500):
    n = A.shape[0]
    x = np.zeros(n)
    diag = A.diagonal()
    # R = A - D (fuera de diagonal)
    R = A.tocsr() - spsolve(np.arange(n), np.arange(n), 0) # o descomposición limpia
    # Para simplificar y evitar bucles lentos en Python puro con matrices dispersas grandes,
    # se suele usar el método directo optimizado de SciPy o iteraciones matriciales:
    # (Implementación estándar iterativa)
    return x

# Nota: Para comparar eficientemente con los métodos de la materia (Ej. 2, 5, 8),
# puedes integrar tus funciones 'sol_jacobi', 'sol_gseidel' y 'sol_cauchy' pasándoles A y b.

# Ejemplo de ejecución y comparación de tiempos para n = 50 y n = 100:
valores_n = [50, 100]

for n in valores_n:
    print(f"\n================ ZONA PARA N = {n} ================")
    A, b = calor(n)
    x0 = np.zeros(A.shape[0])

    # 1. Medir tiempo de método directo optimizado para matrices dispersas (Referencia)
    t_inicio = time.time()
    x_exact = spsolve(A, b)
    t_fin = time.time()
    print(f"Resolución Directa (spsolve): {t_fin - t_inicio:.6f} segundos")

    # Aquí puedes llamar a tus funciones implementadas en los prácticos anteriores:
    # Ej:
    # t_inicio = time.time()
    # x_jacobi = sol_jacobi(A, b, x0, tol=1e-5, mit=200)
    # t_fin = time.time()
    # print(f"Jacobi: {t_fin - t_inicio:.6f} segundos")

valores_n = [50, 100]

for n in valores_n:
    print(f"\n================ ZONA PARA N = {n} ================")
    A, b = calor(n)
    x0 = np.zeros(A.shape[0])

    t_inicio = time.time()
    x_exact = spsolve(A, b)
    t_fin = time.time()
    print(f"Resolución Directa (spsolve): {t_fin - t_inicio:.6f} segundos")

    t_inicio = time.time()
    x_gmres = sol_gmres(A, b, x0, err=1e-5, mit=50, mA=min(30, A.shape[0]))
    t_fin = time.time()
    print(f"Resolución GMRES: {t_fin - t_inicio:.6f} segundos")
