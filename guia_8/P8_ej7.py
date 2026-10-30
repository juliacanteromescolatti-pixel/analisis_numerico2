"""
EJERCICIO 7 - Reglas de busqueda lineal de Armijo y de Goldstein
=================================================================

Se implementan las funciones regla_armijo y regla_goldstein. Ambas reciben

    f     : funcion que retorna f(x), grad f(x), Hess f(x) con la sintaxis del
            ejercicio 6b:  f(x, valfun=..., gradfun=..., hesfun=...)
    x     : punto actual
    d     : direccion de descenso
    phi0  : phi(0)  = f(x)
    dphi0 : phi'(0) = grad f(x)^T d   (debe ser < 0)

y retornan x+ = x + alpha*d, donde alpha cumple la regla correspondiente.

-------------------------------------------------------------------------
IDEA GENERAL
-------------------------------------------------------------------------
Se trabaja con la funcion de una variable
        phi(alpha) = f(x + alpha*d),    alpha >= 0,
con phi(0) = phi0 y phi'(0) = dphi0 = grad f(x)^T d < 0.
Como dphi0 < 0, d es direccion de descenso: para alpha chico, phi(alpha) < phi0.

-------------------------------------------------------------------------
REGLA DE ARMIJO  (condicion de decrecimiento suficiente)
-------------------------------------------------------------------------
    phi(alpha) <= phi0 + c1 * alpha * dphi0,        c1 in (0, 1)  (tipico 1e-4)

La recta phi0 + c1*alpha*dphi0 tiene pendiente c1*dphi0 (menos empinada que la
tangente en 0). Se exige que phi quede por DEBAJO de esa recta.

Algoritmo (backtracking): se parte de alpha = alpha0 (tipico 1) y mientras
no se cumpla la condicion, se reduce alpha <- rho*alpha, con rho in (0, 1)
(tipico 0.5). Como dphi0 < 0, el proceso termina en finitos pasos.

-------------------------------------------------------------------------
REGLA DE GOLDSTEIN  (decrecimiento suficiente + paso no demasiado chico)
-------------------------------------------------------------------------
Con c in (0, 1/2) se piden DOS condiciones:

    (G1)  phi(alpha) <= phi0 + c       * alpha * dphi0     (Armijo)
    (G2)  phi(alpha) >= phi0 + (1 - c) * alpha * dphi0     (no muy chico)

Como dphi0 < 0 y 1-c > c, la recta de (G2) queda por DEBAJO de la de (G1) y
los alpha aceptables son aquellos donde phi esta entre las dos rectas. (G2)
evita pasos demasiado chicos (que Armijo con backtracking si permitiria).

Algoritmo (acotacion + biseccion), manteniendo un intervalo [lo, hi]
(inicialmente lo = 0, hi = +inf) y alpha = alpha0:
    * si falla (G1): alpha es muy grande -> hi = alpha, alpha = (lo + hi)/2
    * si falla (G2): alpha es muy chico  -> lo = alpha,
          y alpha = 2*alpha si hi = +inf  (expansion),
              o   alpha = (lo + hi)/2     si hi es finito (biseccion)
    * si se cumplen ambas: se acepta alpha.

-------------------------------------------------------------------------
"EVITE OPERACIONES INNECESARIAS"
-------------------------------------------------------------------------
  * En cada iteracion solo se evalua phi(alpha) = f(x + alpha d)
    (valfun=True, gradfun=False, hesfun=False): nunca se piden grad ni Hess.
  * phi0 y dphi0 vienen como dato (no se recalculan).
  * Se calcula una sola vez el producto c*dphi0 (y (1-c)*dphi0).
"""

import numpy as np


# ---------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------
def _validar(x, d, dphi0, alpha0):
    """Convierte x, d a arrays y verifica las hipotesis de la busqueda lineal."""
    x = np.asarray(x, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    if x.shape != d.shape:
        raise ValueError("x y d deben tener la misma dimension.")
    if not dphi0 < 0:
        raise ValueError(
            f"d no es direccion de descenso: se requiere phi'(0) < 0 "
            f"(recibido: {dphi0})."
        )
    if not alpha0 > 0:
        raise ValueError("alpha0 debe ser positivo.")
    return x, d


def _phi(f, x, d, alpha):
    """phi(alpha) = f(x + alpha d), pidiendo SOLO el valor de f."""
    return f(x + alpha * d, valfun=True, gradfun=False, hesfun=False)


# ---------------------------------------------------------------------
# Regla de Armijo
# ---------------------------------------------------------------------
def regla_armijo(f, x, d, phi0, dphi0, alpha0=1.0, c1=1e-4, rho=0.5,
                 max_iter=100, ret_alpha=False):
    """
    Busqueda lineal con la regla de Armijo (backtracking).

    Parametros
    ----------
    f      : callable con la sintaxis de fun_mlocal.
    x, d   : punto actual y direccion de descenso.
    phi0   : f(x).
    dphi0  : grad f(x)^T d  (< 0).
    alpha0 : paso inicial (por defecto 1).
    c1     : constante de Armijo en (0, 1) (por defecto 1e-4).
    rho    : factor de reduccion en (0, 1) (por defecto 0.5).
    max_iter : maximo de reducciones.
    ret_alpha : si True, retorna (x_mas, alpha); si False, solo x_mas.

    Retorna
    -------
    x_mas = x + alpha*d  con  phi(alpha) <= phi0 + c1*alpha*dphi0.
    """
    if not 0 < c1 < 1:
        raise ValueError("c1 debe estar en (0, 1).")
    if not 0 < rho < 1:
        raise ValueError("rho debe estar en (0, 1).")
    x, d = _validar(x, d, dphi0, alpha0)

    pend = c1 * dphi0              # pendiente de la recta de Armijo (se calcula 1 vez)
    alpha = alpha0

    for _ in range(max_iter):
        # Condicion de Armijo: phi(alpha) <= phi0 + c1*alpha*dphi0
        if _phi(f, x, d, alpha) <= phi0 + alpha * pend:
            x_mas = x + alpha * d
            return (x_mas, alpha) if ret_alpha else x_mas
        alpha *= rho               # backtracking

    raise RuntimeError(
        f"regla_armijo: no se hallo alpha en {max_iter} iteraciones "
        f"(ultimo alpha = {alpha:.3e})."
    )


# ---------------------------------------------------------------------
# Regla de Goldstein
# ---------------------------------------------------------------------
def regla_goldstein(f, x, d, phi0, dphi0, alpha0=1.0, c=0.25,
                    max_iter=100, ret_alpha=False):
    """
    Busqueda lineal con la regla de Goldstein (acotacion + biseccion).

    Parametros
    ----------
    f, x, d, phi0, dphi0, alpha0, max_iter, ret_alpha : como en regla_armijo.
    c : constante de Goldstein en (0, 1/2) (por defecto 0.25).

    Retorna
    -------
    x_mas = x + alpha*d  con
        phi0 + (1-c)*alpha*dphi0  <=  phi(alpha)  <=  phi0 + c*alpha*dphi0.
    """
    if not 0 < c < 0.5:
        raise ValueError("c debe estar en (0, 1/2).")
    x, d = _validar(x, d, dphi0, alpha0)

    pend_sup = c * dphi0           # pendiente de la recta de (G1)
    pend_inf = (1.0 - c) * dphi0   # pendiente de la recta de (G2)

    lo, hi = 0.0, np.inf           # intervalo que contiene un alpha aceptable
    alpha = alpha0

    for _ in range(max_iter):
        phi = _phi(f, x, d, alpha)

        if phi > phi0 + alpha * pend_sup:
            # (G1) falla: alpha demasiado grande
            hi = alpha
            alpha = 0.5 * (lo + hi)
        elif phi < phi0 + alpha * pend_inf:
            # (G2) falla: alpha demasiado chico
            lo = alpha
            alpha = 2.0 * alpha if np.isinf(hi) else 0.5 * (lo + hi)
        else:
            # Se cumplen (G1) y (G2): alpha aceptado
            x_mas = x + alpha * d
            return (x_mas, alpha) if ret_alpha else x_mas

    raise RuntimeError(
        f"regla_goldstein: no se hallo alpha en {max_iter} iteraciones "
        f"(ultimo alpha = {alpha:.3e})."
    )


# =====================================================================
# PRUEBAS  (usa fun_mlocal del ejercicio 6b)
# =====================================================================
if __name__ == "__main__":
    from ejercicio6b_fun_mlocal import fun_mlocal

    np.set_printoptions(precision=6, suppress=True)

    # Punto de prueba y direccion de maximo descenso d = -grad f(x)
    x = np.array([1.0, 1.0, 0.5])
    f0, g0 = fun_mlocal(x, valfun=True, gradfun=True)
    d = -g0
    phi0 = f0
    dphi0 = g0 @ d                 # = -||grad||^2 < 0
    print(f"x = {x},  phi0 = {phi0:.6f},  dphi0 = {dphi0:.6f}\n")

    # --- Prueba 1: Armijo ---------------------------------------------
    c1 = 1e-4
    x_a, a_a = regla_armijo(fun_mlocal, x, d, phi0, dphi0, c1=c1, ret_alpha=True)
    assert np.allclose(x_a, x + a_a * d)
    assert fun_mlocal(x_a) <= phi0 + c1 * a_a * dphi0
    print(f"Armijo   : alpha = {a_a:.6f}, f(x+) = {fun_mlocal(x_a):.6f} "
          f"<= {phi0 + c1*a_a*dphi0:.6f}   OK")

    # --- Prueba 2: Goldstein ------------------------------------------
    c = 0.25
    x_g, a_g = regla_goldstein(fun_mlocal, x, d, phi0, dphi0, c=c, ret_alpha=True)
    phi_g = fun_mlocal(x_g)
    cota_sup = phi0 + c * a_g * dphi0
    cota_inf = phi0 + (1 - c) * a_g * dphi0
    assert np.allclose(x_g, x + a_g * d)
    assert cota_inf <= phi_g <= cota_sup
    print(f"Goldstein: alpha = {a_g:.6f}, {cota_inf:.6f} <= f(x+) = "
          f"{phi_g:.6f} <= {cota_sup:.6f}   OK")

    # --- Prueba 3: retorno por defecto es solo x+ --------------------
    xp = regla_armijo(fun_mlocal, x, d, phi0, dphi0)
    assert isinstance(xp, np.ndarray) and xp.shape == x.shape
    print("Retorno por defecto (solo x+) .............. OK")

    # --- Prueba 4: Goldstein exige paso mas grande que backtracking ---
    # Cuadratica f(x) = x^2/2 (en 1D), x = 1, d = -1: phi(a) = (1-a)^2/2.
    def cuad(z, valfun=True, gradfun=False, hesfun=False):
        z = np.asarray(z, dtype=float)
        return 0.5 * float(z @ z)

    xq, dq = np.array([1.0]), np.array([-1.0])
    pq, dpq = 0.5, -1.0
    _, a_arm = regla_armijo(cuad, xq, dq, pq, dpq, alpha0=1e-3, ret_alpha=True)
    _, a_gol = regla_goldstein(cuad, xq, dq, pq, dpq, alpha0=1e-3, ret_alpha=True)
    # Armijo acepta el paso inicial diminuto; Goldstein lo agranda (G2)
    assert a_gol > a_arm
    print(f"Paso inicial 1e-3: Armijo acepta alpha = {a_arm:.4g}; "
          f"Goldstein lo expande hasta alpha = {a_gol:.4g}   OK")

    # --- Prueba 5: validaciones ---------------------------------------
    try:
        regla_armijo(fun_mlocal, x, -d, phi0, -dphi0)   # dphi0 > 0
    except ValueError as err:
        print("Direccion de ascenso rechazada ............. OK ->", err)

    # --- Prueba 6: minimizacion completa con cada regla ---------------
    for nombre, regla in (("Armijo", regla_armijo), ("Goldstein", regla_goldstein)):
        z = np.array([1.0, 1.0, 0.5])
        for _ in range(200):
            fz, gz = fun_mlocal(z, valfun=True, gradfun=True)
            if np.linalg.norm(gz) < 1e-6:
                break
            dz = -gz
            z = regla(fun_mlocal, z, dz, fz, gz @ dz)
        assert fun_mlocal(z) < fun_mlocal(np.array([1.0, 1.0, 0.5]))
        print(f"Descenso con {nombre:9s}: f final = {fun_mlocal(z):.3e}, "
              f"||grad|| = {np.linalg.norm(fun_mlocal(z, False, True)):.2e}   OK")

    print("\nTodas las pruebas pasaron.")