"""
EJERCICIO 6 b) - Implementacion de la funcion fun_mlocal
=========================================================

Sea n >= 2 y f : R^n -> R definida por

    f(x) = (1 - x_n)^3 * sum_{i=1}^{n-1} x_i^2  +  x_n^2.

Se pide una funcion fun_mlocal que retorne opcionalmente
    f(x)      en R,
    grad f(x) en R^n,
    Hess f(x) en R^{n x n}.

-------------------------------------------------------------------------
NOTACION
-------------------------------------------------------------------------
    t = 1 - x_n            S = sum_{i=1}^{n-1} x_i^2  (suma de los n-1 primeros)

    f(x) = t^3 * S + x_n^2

-------------------------------------------------------------------------
DERIVACION DEL GRADIENTE
-------------------------------------------------------------------------
Para i = 1, ..., n-1  (S depende de x_i, t no depende de x_i):
    df/dx_i = t^3 * 2*x_i = 2 t^3 x_i

Para i = n  (t depende de x_n con dt/dx_n = -1; S no depende de x_n):
    df/dx_n = 3 t^2 * (-1) * S + 2*x_n = -3 t^2 S + 2 x_n

-------------------------------------------------------------------------
DERIVACION DE LA HESSIANA  (matriz simetrica de derivadas segundas)
-------------------------------------------------------------------------
Para i, j <= n-1:
    d2f/dx_i dx_j = d/dx_j (2 t^3 x_i) = 2 t^3 * delta_ij
    (solo hay diagonal: vale 2 t^3 si i = j y 0 si i != j)

Para i <= n-1, j = n:
    d2f/dx_i dx_n = d/dx_n (2 t^3 x_i) = 2 * 3 t^2 * (-1) * x_i = -6 t^2 x_i

Para i = j = n:
    d2f/dx_n^2 = d/dx_n (-3 t^2 S + 2 x_n)
               = -3 * 2 t * (-1) * S + 2 = 6 t S + 2

Es decir, con x_hat = (x_1, ..., x_{n-1}) en R^{n-1}:

            [ 2 t^3 I_{n-1}        -6 t^2 x_hat ]
    H(x) =  [                                   ]
            [ -6 t^2 x_hat^T        6 t S + 2   ]

-------------------------------------------------------------------------
SINTAXIS DE USO (segun el enunciado)
-------------------------------------------------------------------------
    f          = fun_mlocal(x)                                   # valores por defecto
    f, df      = fun_mlocal(x, valfun=True, gradfun=True)
    f, df, d2f = fun_mlocal(x, valfun=True, gradfun=True, hesfun=True)

Valores por defecto: valfun=True, gradfun=False, hesfun=False.
La funcion devuelve SOLO lo pedido, siempre en el orden (f, df, d2f).
Si se pide una unica cosa se devuelve directamente (sin tupla).

-------------------------------------------------------------------------
"EVITE REALIZAR OPERACIONES INNECESARIAS"
-------------------------------------------------------------------------
  * t y S se calculan UNA sola vez y se reutilizan en f, grad y Hess.
  * grad solo se arma si gradfun=True; Hess solo si hesfun=True.
  * La Hessiana es (casi) diagonal: se arma con np.zeros + asignacion de
    diagonal y ultima fila/columna (sin dobles bucles).
"""

import numpy as np


def fun_mlocal(x, valfun=True, gradfun=False, hesfun=False):
    """
    Evalua f(x), grad f(x) y/o Hess f(x) de
        f(x) = (1 - x_n)^3 * sum_{i<n} x_i^2 + x_n^2.

    Parametros
    ----------
    x : array_like de n >= 2 componentes.
    valfun, gradfun, hesfun : bool
        Indican que calcular (por defecto: True, False, False).

    Retorna
    -------
    Solo lo pedido, en el orden (f, df, d2f):
        f   : float
        df  : ndarray (n,)
        d2f : ndarray (n, n)
    Si se pidio una sola cosa, se retorna directamente (no una tupla).
    """
    # ---------------------------------------------------------------
    # Paso 1: validar y normalizar la entrada
    # ---------------------------------------------------------------
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError(f"Se requiere n >= 2 (recibido: n = {n}).")
    if not (valfun or gradfun or hesfun):
        raise ValueError("Al menos uno de valfun, gradfun, hesfun debe ser True.")

    # ---------------------------------------------------------------
    # Paso 2: cantidades comunes (se calculan una sola vez)
    # ---------------------------------------------------------------
    xhat = x[:-1]            # (x_1, ..., x_{n-1})
    xn = x[-1]               # x_n
    t = 1.0 - xn             # t = 1 - x_n
    S = xhat @ xhat          # S = sum_{i<n} x_i^2
    t2 = t * t               # t^2
    t3 = t2 * t              # t^3

    resultados = []

    # ---------------------------------------------------------------
    # Paso 3: valor de la funcion
    # ---------------------------------------------------------------
    if valfun:
        resultados.append(t3 * S + xn * xn)

    # ---------------------------------------------------------------
    # Paso 4: gradiente
    # ---------------------------------------------------------------
    if gradfun:
        g = np.empty(n)
        g[:-1] = 2.0 * t3 * xhat             # df/dx_i = 2 t^3 x_i,  i < n
        g[-1] = -3.0 * t2 * S + 2.0 * xn     # df/dx_n = -3 t^2 S + 2 x_n
        resultados.append(g)

    # ---------------------------------------------------------------
    # Paso 5: Hessiana
    # ---------------------------------------------------------------
    if hesfun:
        H = np.zeros((n, n))
        idx = np.arange(n - 1)
        H[idx, idx] = 2.0 * t3               # bloque (n-1)x(n-1): 2 t^3 I
        col = -6.0 * t2 * xhat               # d2f/dx_i dx_n = -6 t^2 x_i
        H[:-1, -1] = col                     # ultima columna
        H[-1, :-1] = col                     # ultima fila (simetria)
        H[-1, -1] = 6.0 * t * S + 2.0        # d2f/dx_n^2
        resultados.append(H)

    # ---------------------------------------------------------------
    # Paso 6: devolver lo pedido
    # ---------------------------------------------------------------
    return resultados[0] if len(resultados) == 1 else tuple(resultados)


# =====================================================================
# PRUEBAS
# =====================================================================
def _grad_dif_finitas(x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros(x.size)
    for j in range(x.size):
        d = np.zeros(x.size)
        d[j] = h
        g[j] = (fun_mlocal(x + d) - fun_mlocal(x - d)) / (2 * h)
    return g


def _hess_dif_finitas(x, h=1e-5):
    x = np.asarray(x, dtype=float)
    H = np.zeros((x.size, x.size))
    for j in range(x.size):
        d = np.zeros(x.size)
        d[j] = h
        gp = fun_mlocal(x + d, valfun=False, gradfun=True)
        gm = fun_mlocal(x - d, valfun=False, gradfun=True)
        H[:, j] = (gp - gm) / (2 * h)
    return H


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # --- Prueba 1: el origen (punto estacionario del item a) -----------
    n = 4
    x0 = np.zeros(n)
    f0, g0, H0 = fun_mlocal(x0, valfun=True, gradfun=True, hesfun=True)
    assert f0 == 0.0
    assert np.allclose(g0, 0.0)             # grad f(0) = 0  -> estacionario
    assert np.allclose(H0, 2.0 * np.eye(n))  # Hess f(0) = 2 I -> def. positiva
    print("Prueba 1 (origen: f=0, grad=0, Hess=2I) ........ OK")

    # --- Prueba 2: formas de llamada y valores por defecto -------------
    x = np.array([0.3, -0.7, 1.2, 0.5])
    f_a = fun_mlocal(x)
    f_b, g_b = fun_mlocal(x, valfun=True, gradfun=True)
    f_c, g_c, H_c = fun_mlocal(x, valfun=True, gradfun=True, hesfun=True)
    assert np.isscalar(f_a) or np.ndim(f_a) == 0
    assert np.isclose(f_a, f_b) and np.isclose(f_a, f_c)
    assert np.allclose(g_b, g_c)
    assert g_c.shape == (4,) and H_c.shape == (4, 4)
    print("Prueba 2 (formas de llamada) .................... OK")

    # --- Prueba 3: pedir solo gradiente / solo Hessiana ----------------
    g_solo = fun_mlocal(x, valfun=False, gradfun=True)
    H_solo = fun_mlocal(x, valfun=False, hesfun=True)
    assert np.allclose(g_solo, g_c) and np.allclose(H_solo, H_c)
    print("Prueba 3 (solo grad / solo Hess) ................ OK")

    # --- Prueba 4: derivadas vs diferencias finitas (puntos al azar) ---
    rng = np.random.default_rng(1)
    peor_g, peor_H = 0.0, 0.0
    for _ in range(100):
        xr = rng.uniform(-2, 2, size=rng.integers(2, 7))
        _, g, H = fun_mlocal(xr, valfun=True, gradfun=True, hesfun=True)
        peor_g = max(peor_g, np.max(np.abs(g - _grad_dif_finitas(xr))))
        peor_H = max(peor_H, np.max(np.abs(H - _hess_dif_finitas(xr))))
        assert np.allclose(H, H.T)           # la Hessiana es simetrica
    assert peor_g < 1e-4 and peor_H < 1e-3
    print(f"Prueba 4 (vs dif. finitas, 100 pts) ............. OK "
          f"-> err grad = {peor_g:.1e}, err Hess = {peor_H:.1e}")

    # --- Prueba 5: el origen NO es minimizador global (item a) ---------
    # Tomando x_n = 2 (t = -1, t^3 = -1) y x_1 grande: f = -x_1^2 + 4 < 0 = f(0)
    xg = np.array([10.0, 0.0, 0.0, 2.0])
    assert fun_mlocal(xg) < fun_mlocal(np.zeros(4))
    print(f"Prueba 5 (f(10,0,0,2) = {fun_mlocal(xg):.1f} < f(0) = 0) "
          f"-> no es min. global . OK")

    # --- Prueba 6: entradas invalidas ----------------------------------
    for malo in ([1.0], ):
        try:
            fun_mlocal(malo)
        except ValueError as err:
            print("Prueba 6 (entrada invalida) ..................... OK ->", err)

    print("\nTodas las pruebas pasaron.")