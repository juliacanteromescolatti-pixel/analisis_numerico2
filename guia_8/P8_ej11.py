"""
EJERCICIO 11 - Separacion de dos grupos de datos (regresion logistica)
=======================================================================

ENUNCIADO
---------
Se tienen datos {(x^1,y_1), ..., (x^d,y_d)} con x^k en R^n y rotulos y_k = 1
o y_k = -1. Se busca (w,b) en R^n x R tal que el hiperplano
        H_{w,b} = { x : <w,x> = b }
separe los dos grupos. Para ello se resuelve

    minimizar   sum_{k=1}^{d} log( 1 + exp( -(<w,x^k> - b) * y_k ) ).
   (w,b)

Se pide generar datos en R^2, resolver con los METODOS DEL EJERCICIO 9
(min_cauchy, min_newton y min_bfgs con la regla de Armijo, copiados mas abajo
tal cual) y graficar los datos junto con el hiperplano separador.

-------------------------------------------------------------------------
PLANTEO: una sola variable z = (w, b) en R^{n+1}
-------------------------------------------------------------------------
Se define, para cada dato, el vector
        a_k = y_k * (x^k, -1)  en R^{n+1}      y el margen   m_k = <a_k, z>.
Entonces  m_k = y_k (<w,x^k> - b)  y el problema es

        minimizar  f(z) = sum_k l(m_k),      l(m) = log(1 + e^{-m}).

Si A es la matriz d x (n+1) cuyas filas son a_k:  m = A z.

-------------------------------------------------------------------------
DERIVADAS (paso a paso)
-------------------------------------------------------------------------
Sea s(m) = 1/(1 + e^{m}).  Entonces
    l'(m)  = -e^{-m}/(1 + e^{-m}) = -1/(1 + e^{m}) = -s(m)
    l''(m) = e^{m}/(1 + e^{m})^2  =  s(m) (1 - s(m))  > 0.

Como m_k = <a_k, z> (lineal en z), por la regla de la cadena:
    grad f(z) = sum_k l'(m_k) a_k          = - A^T s,        s = (s(m_1),...,s(m_d))
    Hess f(z) = sum_k l''(m_k) a_k a_k^T   =   A^T D A,      D = diag( s_k (1 - s_k) ).

D tiene entradas positivas, luego Hess f es semidefinida positiva (definida
positiva si A tiene rango n+1): f es convexa (estrictamente convexa) y todo
punto estacionario es el minimizador global.

OBSERVACION IMPORTANTE (existencia de solucion)
-----------------------------------------------
Si los datos fueran perfectamente separables por un hiperplano, f(t z) -> 0
cuando t -> +inf: el infimo es 0 y NO se alcanza (|w| crece sin limite).
Los metodos "convergen" entonces a un w gigante que depende del paso inicial.
Por eso se generan dos nubes con una superposicion leve: asi el problema
tiene minimizador unico y el hiperplano obtenido separa casi todos los datos
(se informa cuantos quedan del lado equivocado).

-------------------------------------------------------------------------
ESTABILIDAD NUMERICA
-------------------------------------------------------------------------
    log(1 + e^{-m}) = np.logaddexp(0, -m)            (no desborda para |m| grande)
    s(m) = 1/(1 + e^{m}) = exp(-np.logaddexp(0, m))
Ademas, m, s y las matrices se calculan una sola vez por evaluacion, y el
gradiente / la Hessiana solo si se piden ("evite operaciones innecesarias").

-------------------------------------------------------------------------
COMO SE CONECTA CON LAS FUNCIONES DEL EJERCICIO 9
-------------------------------------------------------------------------
Los metodos del ejercicio 9 esperan que f(x) devuelva una tupla:
    (f, grad)         para min_cauchy, min_bfgs y para regla_armijo
    (f, grad, hess)   para min_newton
Por eso se definen dos funciones sobre la misma fun_logistica:
    f_grad(z) -> (f, grad)          (no se arma la Hessiana: no hace falta)
    f_hess(z) -> (f, grad, hess)
regla_armijo hace  `fx, grad = f(x)`  y  `f_next, _ = f(x_next)`, es decir
necesita EXACTAMENTE 2 valores; por eso la regla usada siempre evalua f_grad
(tambien dentro de Newton).

-------------------------------------------------------------------------
HIPERPLANO EN R^2 PARA GRAFICAR
-------------------------------------------------------------------------
    w1*x1 + w2*x2 = b   ->   x2 = (b - w1*x1)/w2     (si w2 != 0).
Regla de clasificacion: sign(<w,x> - b).
"""

from functools import partial

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
# EJERCICIO 11: funcion objetivo, datos y programa principal
# =====================================================================
def fun_logistica(z, A, valfun=True, gradfun=False, hesfun=False):
    """
    f(z) = sum_k log(1 + exp(-<a_k, z>)),  con A = filas a_k = y_k (x^k, -1).

    Retorna SOLO lo pedido, en el orden (f, grad, Hess) (si es una sola cosa,
    la devuelve directamente). Valores por defecto: valfun=True, resto False.
    """
    m = A @ z                                   # margenes m_k (una sola vez)
    resultados = []
    if valfun:
        resultados.append(np.sum(np.logaddexp(0.0, -m)))
    if gradfun or hesfun:
        s = np.exp(-np.logaddexp(0.0, m))       # s_k = 1/(1 + e^{m_k})
        if gradfun:
            resultados.append(-A.T @ s)         # grad f = -A^T s
        if hesfun:
            D = s * (1.0 - s)                   # l''(m_k)
            resultados.append(A.T @ (A * D[:, None]))   # A^T diag(D) A
    return resultados[0] if len(resultados) == 1 else tuple(resultados)


def generar_datos(d=100, sep=1.2, std=1.0, semilla=1):
    """
    Genera d puntos en R^2: la mitad con rotulo +1 alrededor de (sep, sep) y
    la otra mitad con rotulo -1 alrededor de (-sep, -sep), con desvio std.
    Retorna X (d x 2), y (d,) con valores +-1.
    """
    rng = np.random.default_rng(semilla)
    d1 = d // 2
    X = np.vstack([rng.normal([sep, sep], std, size=(d1, 2)),
                   rng.normal([-sep, -sep], std, size=(d - d1, 2))])
    y = np.concatenate([np.ones(d1), -np.ones(d - d1)])
    return X, y


def armar_A(X, y):
    """Filas a_k = y_k * (x^k, -1)."""
    return y[:, None] * np.hstack([X, -np.ones((X.shape[0], 1))])


def _errores_clasificacion(X, y, z):
    return int(np.sum(np.sign(X @ z[:2] - z[2]) != y))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=5, suppress=True)

    # ---------------- datos y funciones para los metodos ------------
    X, y = generar_datos(d=100, sep=1.2, std=1.0, semilla=1)
    A = armar_A(X, y)

    def f_grad(z):                      # (f, grad): Cauchy, BFGS y regla_armijo
        return fun_logistica(z, A, valfun=True, gradfun=True, hesfun=False)

    def f_hess(z):                      # (f, grad, hess): Newton
        return fun_logistica(z, A, valfun=True, gradfun=True, hesfun=True)

    def regla(f, x, d):                 # regla(f, x, d) -> alpha (ver cabecera)
        return regla_armijo(f_grad, x, d)

    z0 = np.zeros(3)                    # punto inicial: w = 0, b = 0
    eps = 1e-6

    # ---------------- PRUEBA 1: derivadas vs diferencias finitas ----
    rng = np.random.default_rng(0)
    zt = rng.normal(size=3)
    g, H = fun_logistica(zt, A, valfun=False, gradfun=True, hesfun=True)
    h = 1e-6
    g_fd = np.array([(fun_logistica(zt + h*e, A) - fun_logistica(zt - h*e, A)) / (2*h)
                     for e in np.eye(3)])
    H_fd = np.array([(fun_logistica(zt + h*e, A, valfun=False, gradfun=True)
                      - fun_logistica(zt - h*e, A, valfun=False, gradfun=True)) / (2*h)
                     for e in np.eye(3)]).T
    assert np.allclose(g, g_fd, atol=1e-5) and np.allclose(H, H_fd, atol=1e-5)
    assert np.allclose(H, H.T) and np.all(np.linalg.eigvalsh(H) > 0)
    print("Prueba 1: gradiente y Hessiana coinciden con diferencias finitas;"
          " Hessiana simetrica y definida positiva ... OK")

    # ---------------- resolver con los tres metodos del ejercicio 9 -
    metodos = [("Cauchy", min_cauchy, f_grad, 20000),
               ("Newton", min_newton, f_hess, 500),
               ("BFGS",   min_bfgs,   f_grad, 5000)]
    resultados = {}
    print(f"\nPunto inicial z0 = (w, b) = {z0},  f(z0) = {f_grad(z0)[0]:.6f}  (= d*log 2)\n")
    print(f"{'metodo':8s} {'iter':>5s} {'convergio':>10s} {'f(z*)':>12s} {'||grad||':>10s}"
          f" {'mal clasif.':>12s}   (w1, w2, b)")
    for nombre, metodo, f, m in metodos:
        z, k, hist = metodo(f, z0, eps, m, regla)       # (x, iteraciones, historial)
        fz, gz = f_grad(z)
        resultados[nombre] = {"z": z, "k": k, "hist": hist}
        print(f"{nombre:8s} {k:5d} {str(k < m):>10s} {fz:12.6f} "
              f"{np.linalg.norm(gz):10.2e} {_errores_clasificacion(X, y, z):12d}   {z}")
        assert k < m                                     # converge antes de m

    # ---------------- PRUEBA 2: los tres metodos coinciden ----------
    z_cauchy = resultados["Cauchy"]["z"]
    for nombre in ("Newton", "BFGS"):
        assert np.allclose(resultados[nombre]["z"], z_cauchy, atol=1e-4)
    zN = resultados["Newton"]["z"]
    gN, HN = fun_logistica(zN, A, valfun=False, gradfun=True, hesfun=True)
    assert np.linalg.norm(gN) < 1e-6 and np.all(np.linalg.eigvalsh(HN) > 0)
    print("\nPrueba 2: los tres metodos hallan el mismo (w,b); en la solucion"
          " grad = 0 y Hess definida positiva ... OK")

    # ---------------- grafico ---------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.scatter(*X[y == 1].T, c="tab:blue", marker="o", edgecolor="k", label="grupo  y = +1")
    ax.scatter(*X[y == -1].T, c="tab:red", marker="s", edgecolor="k", label="grupo  y = -1")
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    estilos = {"Cauchy": ("-", 3.5, "k"), "Newton": ("--", 2.2, "tab:green"),
               "BFGS": (":", 2.2, "tab:orange")}
    for nombre, res in resultados.items():
        w1, w2, b = res["z"]
        ls, lw, col = estilos[nombre]
        ax.plot(x1, (b - w1 * x1) / w2, ls=ls, lw=lw, color=col,
                label=f"hiperplano {nombre}")
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(r"Datos e hiperplano separador $\langle w,x\rangle = b$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.grid(alpha=0.3)

    for nombre, res in resultados.items():              # convergencia (usa el historial)
        ng = [np.linalg.norm(f_grad(z)[1]) for z in res["hist"]]
        ax2.semilogy(ng, color=estilos[nombre][2], label=nombre, lw=2)
    ax2.axhline(eps, color="gray", ls="--", lw=1)
    ax2.set_xlabel("iteracion"); ax2.set_ylabel(r"$\|\nabla f(z_k)\|_2$")
    ax2.set_title("Convergencia de los tres metodos")
    ax2.legend(); ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("ejercicio11_resultado.png", dpi=150)
    print("Grafico guardado en ejercicio11_resultado.png")
    plt.show()