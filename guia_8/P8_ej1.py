"""
EJERCICIO 1 - Implementacion de la funcion fun_tres
===================================================

Se pide implementar una funcion fun_tres que evalue

    F : R^3 -> R^3,     F(x) = [ F1(x) ]
                               [ F2(x) ]
                               [ F3(x) ]

    F1(x) = x1 + x1*exp(-x1) + 3*x2 + sin(x3) - 3
    F2(x) = x1^2 + x2^2 + x3^2 - 1
    F3(x) = x1^2 + 2*x2*exp(-x1) + 2*x3 - 2

y, opcionalmente, su derivada (matriz jacobiana) F'(x) en R^{3x3}, donde

    F'(x)[i, j] = dFi / dxj.

-------------------------------------------------------------------------
DERIVACION DE LA JACOBIANA (derivada parcial por derivada parcial)
-------------------------------------------------------------------------
Fila 1 (F1 = x1 + x1*exp(-x1) + 3*x2 + sin(x3) - 3):
    dF1/dx1 = 1 + [exp(-x1) + x1*(-exp(-x1))]      (regla del producto)
            = 1 + exp(-x1) - x1*exp(-x1)
            = 1 + (1 - x1)*exp(-x1)
    dF1/dx2 = 3
    dF1/dx3 = cos(x3)

Fila 2 (F2 = x1^2 + x2^2 + x3^2 - 1):
    dF2/dx1 = 2*x1
    dF2/dx2 = 2*x2
    dF2/dx3 = 2*x3

Fila 3 (F3 = x1^2 + 2*x2*exp(-x1) + 2*x3 - 2):
    dF3/dx1 = 2*x1 + 2*x2*(-exp(-x1)) = 2*x1 - 2*x2*exp(-x1)
    dF3/dx2 = 2*exp(-x1)
    dF3/dx3 = 2

Por lo tanto:

            [ 1+(1-x1)e^{-x1}     3          cos(x3) ]
    F'(x) = [ 2*x1               2*x2        2*x3    ]
            [ 2*x1-2*x2*e^{-x1}  2*e^{-x1}   2       ]

-------------------------------------------------------------------------
SINTAXIS DE USO (segun el enunciado)
-------------------------------------------------------------------------
    F     = fun_tres(x)                                  # solo F(x)
    F     = fun_tres(x, valfun=True, derfun=False)       # solo F(x)
    _, DF = fun_tres(x, valfun=False, derfun=True)       # solo F'(x)
    F, DF = fun_tres(x, valfun=True,  derfun=True)       # ambas

-------------------------------------------------------------------------
"EVITE REALIZAR OPERACIONES INNECESARIAS"
-------------------------------------------------------------------------
  * exp(-x1) aparece en F y en F': se calcula UNA sola vez y se reutiliza.
  * F solo se arma si valfun=True.
  * F' solo se arma si derfun=True.
  * Se guardan x1, x2, x3 en variables locales (no se indexa el vector
    repetidas veces).
"""

import numpy as np


def fun_tres(x, valfun=True, derfun=False):
    """
    Evalua F(x) y/o F'(x) para la funcion F: R^3 -> R^3 del ejercicio 1.

    Parametros
    ----------
    x : array_like de 3 componentes
        Punto donde se evalua (lista, tupla o ndarray; fila o columna).
    valfun : bool, por defecto True
        Si es True se calcula F(x).
    derfun : bool, por defecto False
        Si es True se calcula la derivada F'(x) (matriz 3x3).

    Retorna
    -------
    Si derfun es False:
        F : ndarray de forma (3,)
    Si derfun es True:
        (F, DF) : F es ndarray (3,) o None (si valfun=False),
                  DF es ndarray (3, 3).
    """
    # ---------------------------------------------------------------
    # Paso 1: validar y normalizar la entrada
    # ---------------------------------------------------------------
    x = np.asarray(x, dtype=float).ravel()      # vector plano de floats
    if x.size != 3:
        raise ValueError(
            f"x debe tener exactamente 3 componentes (recibido: {x.size})."
        )

    # Si no se pide nada, no hay nada que calcular
    if not valfun and not derfun:
        raise ValueError("Al menos uno de valfun o derfun debe ser True.")

    # ---------------------------------------------------------------
    # Paso 2: componentes y cantidad comun (calculada una sola vez)
    # ---------------------------------------------------------------
    x1, x2, x3 = x
    e = np.exp(-x1)            # exp(-x1) se usa en F1, F3 y en F'

    # ---------------------------------------------------------------
    # Paso 3: evaluar F(x) solo si se pidio
    # ---------------------------------------------------------------
    F = None
    if valfun:
        F = np.array([
            x1 + x1 * e + 3.0 * x2 + np.sin(x3) - 3.0,   # F1
            x1**2 + x2**2 + x3**2 - 1.0,                 # F2
            x1**2 + 2.0 * x2 * e + 2.0 * x3 - 2.0,       # F3
        ])

    # Si solo se pidio F, se devuelve F sola (F = fun_tres(x))
    if not derfun:
        return F

    # ---------------------------------------------------------------
    # Paso 4: evaluar la jacobiana F'(x) (derfun=True)
    # ---------------------------------------------------------------
    DF = np.array([
        [1.0 + (1.0 - x1) * e,    3.0,         np.cos(x3)],   # dF1/dxj
        [2.0 * x1,                2.0 * x2,    2.0 * x3  ],   # dF2/dxj
        [2.0 * x1 - 2.0 * x2 * e, 2.0 * e,     2.0       ],   # dF3/dxj
    ])

    return F, DF


# =====================================================================
# PRUEBAS
# =====================================================================
def _jacobiana_diferencias_finitas(x, h=1e-6):
    """Aproxima F'(x) columna a columna con diferencias centradas."""
    x = np.asarray(x, dtype=float).ravel()
    J = np.zeros((3, 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = h
        J[:, j] = (fun_tres(x + d) - fun_tres(x - d)) / (2.0 * h)
    return J


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # --- Prueba 1: valor en un punto conocido --------------------------
    # En x = (0, 0, 0): F1 = 0+0+0+0-3 = -3, F2 = -1, F3 = 0+0+0-2 = -2
    F = fun_tres([0.0, 0.0, 0.0])
    assert np.allclose(F, [-3.0, -1.0, -2.0])
    print("Prueba 1 (F en el origen) .......... OK ->", F)

    # --- Prueba 2: jacobiana en un punto conocido ----------------------
    # En x = (0, 0, 0): e = 1
    #   [1+1  3  cos0]   [2 3 1]
    #   [0    0  0   ] = [0 0 0]
    #   [0    2  2   ]   [0 2 2]
    _, DF = fun_tres([0.0, 0.0, 0.0], valfun=False, derfun=True)
    assert np.allclose(DF, [[2, 3, 1], [0, 0, 0], [0, 2, 2]])
    print("Prueba 2 (F' en el origen) ......... OK ->\n", DF)

    # --- Prueba 3: las cuatro formas de llamada ------------------------
    x = np.array([0.5, -1.0, 2.0])
    F_a = fun_tres(x)
    F_b = fun_tres(x, valfun=True, derfun=False)
    _, DF_c = fun_tres(x, valfun=False, derfun=True)
    F_d, DF_d = fun_tres(x, valfun=True, derfun=True)
    assert np.allclose(F_a, F_b) and np.allclose(F_a, F_d)
    assert np.allclose(DF_c, DF_d)
    assert F_a.shape == (3,) and DF_d.shape == (3, 3)
    print("Prueba 3 (formas de llamada) ....... OK")

    # --- Prueba 4: con valfun=False, F no se calcula -------------------
    F_none, _ = fun_tres(x, valfun=False, derfun=True)
    assert F_none is None
    print("Prueba 4 (F=None si valfun=False) .. OK")

    # --- Prueba 5: derivada vs diferencias finitas en puntos al azar ---
    rng = np.random.default_rng(0)
    peor = 0.0
    for _ in range(100):
        xr = rng.uniform(-3, 3, size=3)
        _, DFa = fun_tres(xr, valfun=False, derfun=True)
        DFn = _jacobiana_diferencias_finitas(xr)
        peor = max(peor, np.max(np.abs(DFa - DFn)))
    assert peor < 1e-5
    print(f"Prueba 5 (vs dif. finitas, 100 pts)  OK -> error maximo = {peor:.2e}")

    # --- Prueba 6: entrada invalida ------------------------------------
    try:
        fun_tres([1.0, 2.0])
    except ValueError as err:
        print("Prueba 6 (entrada invalida) ........ OK ->", err)

    print("\nTodas las pruebas pasaron.")