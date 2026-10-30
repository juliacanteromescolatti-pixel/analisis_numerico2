"""
EJERCICIO 12 - Estrategia optima de inversion en la cosecha de mani
====================================================================

ENUNCIADO
---------
Un productor cosecha x_i toneladas de mani, almacena (1 - u_i) x_i toneladas
e invierte u_i x_i. La proxima cosecha sera
        x_{i+1} = x_i + w u_i x_i = (1 + w u_i) x_i,       w > 0.
Se busca u_0, ..., u_{N-1} que, partiendo de x_0 = 3.5 toneladas, maximice lo
almacenado en N = 10 cosechas penalizando el incumplimiento de u_i en [0,1]:

    maximizar   J(u) = x_N + sum_{i=0}^{N-1} (1 - u_i) x_i - sum_{i=0}^{N-1} psi(u_i),

    psi(t) = rho * ( min{ t, max{0, t-1} } )^4.

Se resuelve con w = 0.2, rho = 1e3 y rho = 1e5, usando los tres metodos del
ejercicio 9 (min_cauchy, min_newton y min_bfgs con regla de Armijo, copiados
mas abajo tal cual). Como los metodos MINIMIZAN, se minimiza
        f(u) = -J(u).

-------------------------------------------------------------------------
LA PENALIZACION psi
-------------------------------------------------------------------------
Sea  s(t) = min{ t, max{0, t-1} }. Entonces
    t < 0      ->  max{0,t-1} = 0   ->  s = min{t,0} = t       -> psi = rho t^4
    0 <= t <= 1 ->  max{0,t-1} = 0   ->  s = min{t,0} = 0       -> psi = 0
    t > 1      ->  max{0,t-1} = t-1 ->  s = min{t,t-1} = t-1    -> psi = rho (t-1)^4
Fuera de [0,1], ds/dt = 1; dentro, s = 0. Por lo tanto (para todo t):
    psi(t)   = rho s^4,     psi'(t) = 4 rho s^3,     psi''(t) = 12 rho s^2
(psi es de clase C^2). Penaliza violar u >= 0 o u <= 1 y vale 0 si u esta en [0,1].

-------------------------------------------------------------------------
x_i EN FUNCION DE u
-------------------------------------------------------------------------
Con q_i = 1 + w u_i:    x_i = x_0 * q_0 q_1 ... q_{i-1},   i = 0, ..., N.
Derivadas (solo dependen de u_j los x_i con i > j, y q_j aparece una vez):
    dx_i/du_j = w x_i / q_j                  (j < i;  0 si j >= i)
    d2x_i/du_j du_k = w^2 x_i /(q_j q_k)     (j != k, j,k < i;  0 si j = k o max(j,k) >= i)

-------------------------------------------------------------------------
GRADIENTE Y HESSIANA DE J
-------------------------------------------------------------------------
Escribimos J = sum_{i=0}^{N} g_i x_i - sum psi(u_i), con g_i = 1 - u_i (i < N) y
g_N = 1. Sea
        T_j = sum_{i=j+1}^{N} g_i x_i          (suma de "cola", j = 0..N-1).

Gradiente (j = 0..N-1):
    dJ/du_j = sum_{i>j} g_i dx_i/du_j  -  x_j  -  psi'(u_j)
            = w T_j / q_j - x_j - psi'(u_j).
   (el termino -x_j viene de derivar g_j = 1 - u_j en el sumando g_j x_j.)

Hessiana (j, k = 0..N-1):
    d2J/du_j du_k = - [k > j] w x_k / q_j  - [j > k] w x_j / q_k
                    + [j != k] w^2 T_max(j,k) / (q_j q_k)  -  psi''(u_j) [j = k]
  o, equivalentemente, para j != k:
    H_jk = w^2 T_max / (q_j q_k)  -  w x_max / q_min ,   H_jj = - psi''(u_j),
  con max = max(j,k), min = min(j,k). Es simetrica.
Para f = -J:   grad f = -grad J,   Hess f = -Hess J.

Nota: dentro de [0,1] la diagonal de Hess J es 0 (psi'' = 0) pero hay terminos
fuera de la diagonal: la Hessiana es INDEFINIDA. Entonces min_newton
(que resuelve H d = -grad sin correccion) NO puede usarse con la Hessiana
exacta: con H indefinida la direccion no es de descenso y el metodo se
estanca (se comprobo: 20000 iteraciones sin progreso). La solucion, sin tocar
min_newton, es que la funcion objetivo que se le pasa devuelva la Hessiana
CORREGIDA  H + gamma*I  (gamma >= 0 minimo, probando 0, beta, 2beta, 4beta...
hasta que H + gamma*I sea definida positiva, verificado con Cholesky). Es la
correccion E = gamma*I de las notas (metodo cuasi-Newton).

-------------------------------------------------------------------------
COMO SE CONECTA CON LAS FUNCIONES DEL EJERCICIO 9
-------------------------------------------------------------------------
    f_grad(u) -> (f, grad)          para min_cauchy, min_bfgs y regla_armijo
    f_hess(u) -> (f, grad, H+gamma I)   para min_newton
regla_armijo necesita exactamente 2 valores de f(x) (`fx, grad = f(x)`), por
eso la regla usada siempre evalua f_grad (tambien dentro de Newton).

-------------------------------------------------------------------------
CHEQUEO POR FUERZA BRUTA (solucion "bang-bang")
-------------------------------------------------------------------------
Invertir u x_i cuesta 1 tonelada por tonelada y rinde w por cosecha restante;
conviene invertir mientras w (N - i) > 1, o sea en las primeras cosechas y
luego almacenar todo. Se verifica probando los 2^10 vectores u en {0,1}^10.
Para rho grande la solucion penalizada se acerca a esa estrategia (u ~ 1 al
principio, u ~ 0 al final); para rho finito se pasa un poco de [0,1] porque
el costo de la penalizacion es menor que la ganancia de excederse.
"""

from functools import partial
from itertools import product

import numpy as np


# =====================================================================
# FUNCIONES DEL EJERCICIO 9 (copiadas tal cual, sin modificar)
# =====================================================================
# =============================================================================
# Regla de Búsqueda Lineal de Armijo (Ejemplo de función `regla`)
# =============================================================================
def regla_armijo(f, x, d, alpha0=1.0, c1=1e-4, rho=0.5):
    """
    Búsqueda lineal que satisface la condición de Armijo.
    """
    alpha = alpha0
    fx, grad = f(x)

    while True:
        x_next = x + alpha * d
        f_next, _ = f(x_next)
        if f_next <= fx + c1 * alpha * np.dot(grad, d):
            break
        alpha *= rho
        if alpha < 1e-12:
            break
    return alpha


# =============================================================================
# 1. Método de Cauchy (Máximo Descenso)
# =============================================================================
def min_cauchy(f, x0, eps, m, regla):
    """
    Minimización por el Método de Cauchy.

    Parámetros:
    -----------
    f     : función que retorna (f(x), grad_f(x)) o (f(x), grad_f(x), hess_f(x))
    x0    : punto inicial (ndarray)
    eps   : tolerancia del error epsilon
    m     : cantidad máxima de iteraciones
    regla : función de búsqueda lineal regla(f, x, d) -> alpha
    """
    x = np.array(x0, dtype=float)
    historial = [x.copy()]

    for k in range(m):
        _, grad = f(x)[:2]

        # Criterio de parada: ||grad f(x)||_2 < eps
        if np.linalg.norm(grad, 2) < eps:
            return x, k, historial

        # Dirección de Cauchy
        d = -grad

        # Tamaño de paso usando la regla dada
        alpha = regla(f, x, d)

        # Actualización
        x = x + alpha * d
        historial.append(x.copy())

    return x, m, historial


# =============================================================================
# 2. Método de Newton
# =============================================================================
def min_newton(f, x0, eps, m, regla):
    """
    Minimización por el Método de Newton Exacto.
    Se asume que f(x) retorna (f_val, grad, hess).
    """
    x = np.array(x0, dtype=float)
    historial = [x.copy()]

    for k in range(m):
        res = f(x)
        grad = res[1]
        hess = res[2]

        if np.linalg.norm(grad, 2) < eps:
            return x, k, historial

        # Resolver sistema Hessian * d = -grad
        d = np.linalg.solve(hess, -grad)

        # Tamaño de paso usando la regla dada
        alpha = regla(f, x, d)

        # Actualización
        x = x + alpha * d
        historial.append(x.copy())

    return x, m, historial


# =============================================================================
# 3. Método BFGS
# =============================================================================
def min_bfgs(f, x0, eps, m, regla):
    """
    Minimización por el Método BFGS (Cuasi-Newton).
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)  # Aproximación inicial de la Hessiana inversa (Matriz Identidad)
    historial = [x.copy()]

    _, grad = f(x)[:2]

    for k in range(m):
        if np.linalg.norm(grad, 2) < eps:
            return x, k, historial

        # Dirección de descenso d = -H * grad
        d = -H @ grad

        # Tamaño de paso
        alpha = regla(f, x, d)

        # Actualización de la posición
        p = alpha * d
        x_next = x + p

        # Evaluación del nuevo gradiente
        _, grad_next = f(x_next)[:2]
        q = grad_next - grad

        # Actualización de la matriz H (Fórmula BFGS para la inversa)
        pq = np.dot(p, q)
        if pq > 1e-10:  # Condición de curvatura para mantener H definida positiva
            rho = 1.0 / pq
            I = np.eye(n)
            H = (I - rho * np.outer(p, q)) @ H @ (I - rho * np.outer(q, p)) + rho * np.outer(p, p)

        x = x_next
        grad = grad_next
        historial.append(x.copy())

    return x, m, historial


# =====================================================================
# EJERCICIO 12: funcion objetivo y programa principal
# =====================================================================
def fun_cosecha(u, x0=3.5, omega=0.2, rho=1e3,
                valfun=True, gradfun=False, hesfun=False):
    """
    f(u) = -J(u), con
        J(u) = x_N + sum_{i<N} (1-u_i) x_i - sum_{i<N} psi(u_i).

    Retorna SOLO lo pedido, en el orden (f, grad, Hess) (si es una sola cosa,
    la devuelve directamente). Valores por defecto: valfun=True, resto False.
    """
    u = np.asarray(u, dtype=float)
    N = u.size

    # --- cantidades comunes (una sola vez) -------------------------
    q = 1.0 + omega * u                                   # q_i = 1 + w u_i
    x = x0 * np.concatenate(([1.0], np.cumprod(q)))       # x_0, ..., x_N
    s = np.minimum(u, np.maximum(0.0, u - 1.0))           # s(t) = min{t, max{0,t-1}}
    gx = np.concatenate(((1.0 - u) * x[:N], [x[N]]))      # g_i x_i, i = 0..N

    resultados = []
    if valfun:
        J = gx.sum() - rho * np.sum(s**4)                 # x_N + sum (1-u_i)x_i - sum psi
        resultados.append(-J)

    if gradfun or hesfun:
        T = np.cumsum(gx[::-1])[::-1][1:]                 # T_j = sum_{i>j} g_i x_i
        if gradfun:
            gJ = omega * T / q - x[:N] - 4.0 * rho * s**3
            resultados.append(-gJ)
        if hesfun:
            idx = np.arange(N)
            Mx = np.maximum(idx[:, None], idx[None, :])   # max(j,k)
            mn = np.minimum(idx[:, None], idx[None, :])   # min(j,k)
            HJ = omega**2 * T[Mx] / np.outer(q, q) - omega * x[Mx] / q[mn]
            HJ[idx, idx] = -12.0 * rho * s**2             # diagonal: -psi''(u_j)
            resultados.append(-HJ)

    return resultados[0] if len(resultados) == 1 else tuple(resultados)


def objetivo_J(u, **kw):
    """J(u) (a MAXIMIZAR) = -f(u)."""
    return -fun_cosecha(u, **kw)


def mejor_bang_bang(N=10, **kw):
    """Fuerza bruta sobre u in {0,1}^N (psi = 0 ahi). Retorna (J*, u*)."""
    mejor = (-np.inf, None)
    for c in product([0.0, 1.0], repeat=N):
        J = objetivo_J(np.array(c), **kw)
        if J > mejor[0]:
            mejor = (J, np.array(c))
    return mejor


def hessiana_corregida(H, beta=1e-3, max_dup=60):
    """
    Retorna H + gamma*I definida positiva, con gamma = 0 si H ya lo es; si no,
    se prueba gamma = beta - min(diag H) y luego se duplica hasta que la
    factorizacion de Cholesky de H + gamma*I tenga exito.
    """
    n = H.shape[0]
    gamma = 0.0 if np.min(np.diag(H)) > 0 else beta - np.min(np.diag(H))
    for _ in range(max_dup):
        Hc = H + gamma * np.eye(n)
        try:
            np.linalg.cholesky(Hc)              # falla si Hc no es definida positiva
            return Hc
        except np.linalg.LinAlgError:
            gamma = max(2.0 * gamma, beta)
    raise RuntimeError("hessiana_corregida: no se logro una matriz definida positiva.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    N, x0, omega = 10, 3.5, 0.2
    eps = 1e-5
    u0 = 0.5 * np.ones(N)                       # punto inicial (dentro de [0,1])

    # ---------------- PRUEBA 1: psi y su forma por tramos -----------
    t = np.array([-0.3, 0.0, 0.4, 1.0, 1.3])
    s_t = np.minimum(t, np.maximum(0.0, t - 1.0))
    psi_formula = 1e3 * s_t**4
    psi_tramos = 1e3 * np.where(t < 0, t**4, np.where(t > 1, (t - 1)**4, 0.0))
    assert np.allclose(psi_formula, psi_tramos)
    print("Prueba 1: psi(t) = rho*(min{t,max{0,t-1}})^4 coincide con la forma por tramos ... OK")

    # ---------------- PRUEBA 2: derivadas vs diferencias finitas ---
    rng = np.random.default_rng(0)
    for rho in (1e3, 1e5):
        f = partial(fun_cosecha, x0=x0, omega=omega, rho=rho)
        ut = rng.uniform(-0.5, 1.5, N)          # incluye tramos fuera de [0,1]
        g, H = f(ut, valfun=False, gradfun=True, hesfun=True)
        h = 1e-6
        g_fd = np.array([(f(ut + h*e) - f(ut - h*e)) / (2*h) for e in np.eye(N)])
        H_fd = np.array([(f(ut + h*e, valfun=False, gradfun=True)
                          - f(ut - h*e, valfun=False, gradfun=True)) / (2*h)
                         for e in np.eye(N)]).T
        err_g = np.max(np.abs(g - g_fd)) / np.max(np.abs(g))
        err_H = np.max(np.abs(H - H_fd)) / np.max(np.abs(H))
        assert err_g < 1e-7 and err_H < 1e-7 and np.allclose(H, H.T)
        print(f"Prueba 2 (rho={rho:.0e}): errores relativos vs dif. finitas: "
              f"grad {err_g:.1e}, Hess {err_H:.1e}; Hess simetrica ... OK")

    # ---------------- referencia por fuerza bruta -------------------
    J_bb, u_bb = mejor_bang_bang(N, x0=x0, omega=omega, rho=1e3)
    print(f"\nMejor estrategia con u en {{0,1}} (fuerza bruta): u = {u_bb.astype(int)},"
          f" J = {J_bb:.5f}")

    # ---------------- resolver con los tres metodos del ejercicio 9 -
    sol = {}                                    # sol[(rho, metodo)] = (u, k, historial)
    for rho in (1e3, 1e5):
        # Funciones con la convencion del ejercicio 9 (se crean para este rho)
        def f_grad(u, rho=rho):                 # (f, grad): Cauchy, BFGS y regla_armijo
            return fun_cosecha(u, x0=x0, omega=omega, rho=rho,
                               valfun=True, gradfun=True, hesfun=False)

        def f_hess(u, rho=rho):                 # (f, grad, H + gamma I): Newton
            fv, gv, Hv = fun_cosecha(u, x0=x0, omega=omega, rho=rho,
                                     valfun=True, gradfun=True, hesfun=True)
            return fv, gv, hessiana_corregida(Hv)

        def regla(f, x, d, f_grad=f_grad):      # regla(f, x, d) -> alpha
            return regla_armijo(f_grad, x, d)

        metodos = [("Cauchy", min_cauchy, f_grad, 20000),
                   ("Newton", min_newton, f_hess, 500),
                   ("BFGS",   min_bfgs,   f_grad, 5000)]
        print(f"\n===== rho = {rho:.0e}  (w = {omega}, x0 = {x0}, N = {N}) =====")
        print(f"{'metodo':8s} {'iter':>6s} {'convergio':>10s} {'J(u*)':>10s} {'||grad||':>10s}")
        for nombre, metodo, f, m in metodos:
            u, k, hist = metodo(f, u0, eps, m, regla)    # (x, iteraciones, historial)
            fu, gu = f_grad(u)
            sol[(rho, nombre)] = (u, k, hist)
            print(f"{nombre:8s} {k:6d} {str(k < m):>10s} {-fu:10.5f} {np.linalg.norm(gu):10.2e}")
            assert k < m
        uN = sol[(rho, "Newton")][0]
        for nombre in ("Cauchy", "BFGS"):
            assert np.allclose(sol[(rho, nombre)][0], uN, atol=1e-3)
        print("u* (Newton) =", uN)
        print(f"  mayor violacion de [0,1]: {max(np.max(-uN), np.max(uN - 1)):.4f}")
        assert -f_grad(uN)[0] >= J_bb - 1e-8     # penalizado >= mejor estrategia factible
        gN, HN = fun_cosecha(uN, x0=x0, omega=omega, rho=rho,
                             valfun=False, gradfun=True, hesfun=True)
        print(f"  autovalores extremos de Hess f(u*): {np.linalg.eigvalsh(HN)[[0, -1]]}")
        assert np.all(np.linalg.eigvalsh(HN) > 0)    # minimizador local estricto

    print("\nPrueba 3: Cauchy, Newton y BFGS coinciden; J penalizado >= J de la mejor"
          " estrategia {0,1}^N; Hess f(u*) definida positiva ... OK")

    # ---------------- grafico ---------------------------------------
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6))
    ancho = 0.38
    i = np.arange(N)
    for k, rho in enumerate((1e3, 1e5)):
        uN = sol[(rho, "Newton")][0]
        ejes[0].bar(i + (k - 0.5) * ancho, uN, ancho, label=rf"$\rho = 10^{{{int(np.log10(rho))}}}$")
    ejes[0].axhline(0, color="k", lw=0.8); ejes[0].axhline(1, color="k", lw=0.8)
    ejes[0].set_xlabel("cosecha $i$"); ejes[0].set_ylabel("$u_i$")
    ejes[0].set_title("Estrategia optima $u_i$ (Newton)")
    ejes[0].set_xticks(i); ejes[0].legend(); ejes[0].grid(alpha=0.3)

    for rho in (1e3, 1e5):
        uN = sol[(rho, "Newton")][0]
        xs = x0 * np.concatenate(([1.0], np.cumprod(1 + omega * uN)))
        ejes[1].plot(np.arange(N + 1), xs, "o-", label=rf"$\rho = 10^{{{int(np.log10(rho))}}}$")
    ejes[1].set_xlabel("cosecha $i$"); ejes[1].set_ylabel("$x_i$ (toneladas)")
    ejes[1].set_title("Produccion $x_i$ con la estrategia optima")
    ejes[1].legend(); ejes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("ejercicio12_resultado.png", dpi=150)
    print("Grafico guardado en ejercicio12_resultado.png")
    plt.show()