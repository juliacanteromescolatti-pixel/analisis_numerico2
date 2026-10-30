import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# FUNCIONES DEL EJERCICIO 9: Métodos de Minimización No Lineal
# =============================================================================

def min_cauchy(f, x0, eps, m, regla):
    """
    Método de Cauchy (Máximo Descenso) del Ejercicio 9.
    """
    x = np.array(x0, dtype=float)
    historial = [x.copy()]
    
    for k in range(m):
        _, grad = f(x)[:2]
        
        # Criterio de parada sobre la norma del gradiente
        if np.linalg.norm(grad, 2) < eps:
            return x, k, historial
            
        # Dirección de Cauchy: d = -grad f(x)
        d = -grad
        
        # Tamaño de paso determinado por la función de regla
        alpha = regla(f, x, d)
        
        # Actualización de la posición
        x = x + alpha * d
        historial.append(x.copy())
        
    return x, m, historial


def min_newton(f, x0, eps, m, regla):
    """
    Método de Newton del Ejercicio 9.
    """
    x = np.array(x0, dtype=float)
    historial = [x.copy()]
    
    for k in range(m):
        res = f(x)
        grad, hess = res[1], res[2]
        
        if np.linalg.norm(grad, 2) < eps:
            return x, k, historial
            
        # Resolver el sistema lineal para la dirección de Newton: Hess * d = -grad
        try:
            d = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            d = -grad  # Resguardo por si el Hessiano es singular
            
        # Tamaño de paso determinado por la regla
        alpha = regla(f, x, d)
        
        # Actualización
        x = x + alpha * d
        historial.append(x.copy())
        
    return x, m, historial


def min_bfgs(f, x0, eps, m, regla):
    """
    Método BFGS (Cuasi-Newton) del Ejercicio 9.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)  # Aproximación inicial de la matriz inversa del Hessiano
    historial = [x.copy()]
    
    _, grad = f(x)[:2]
    
    for k in range(m):
        if np.linalg.norm(grad, 2) < eps:
            return x, k, historial
            
        # Dirección de descenso: d = -H * grad
        d = -H @ grad
        
        # Tamaño de paso
        alpha = regla(f, x, d)
        
        # Actualización de posición
        p = alpha * d
        x_next = x + p
        
        # Evaluación del nuevo gradiente
        _, grad_next = f(x_next)[:2]
        q = grad_next - grad
        
        # Actualización de la matriz H usando la fórmula BFGS
        pq = np.dot(p, q)
        if pq > 1e-10:
            rho_val = 1.0 / pq
            I = np.eye(n)
            H = (I - rho_val * np.outer(p, q)) @ H @ (I - rho_val * np.outer(q, p)) + rho_val * np.outer(p, p)
            
        x = x_next
        grad = grad_next
        historial.append(x.copy())
        
    return x, m, historial


# =============================================================================
# REGLA DE BÚSQUEDA LINEAL (Condición de Armijo)
# =============================================================================

def regla_armijo(f, x, d, alpha0=1.0, c1=1e-4, rho=0.5):
    """
    Implementación del parámetro 'regla' mediante Búsqueda de Armijo.
    """
    alpha = alpha0
    fx, grad = f(x)[:2]
    
    while True:
        x_next = x + alpha * d
        f_next, _ = f(x_next)[:2]
        if f_next <= fx + c1 * alpha * np.dot(grad, d):
            break
        alpha *= rho
        if alpha < 1e-12:
            break
            
    return alpha


# =============================================================================
# EJERCICIO 10: Prueba en la Función de Rosenbrock
# =============================================================================

def rosenbrock(v):
    """
    Función de Rosenbrock f(x, y) = 100*(y - x^2)^2 + (1 - x)^2.
    Retorna: (f, grad_f, hess_f)
    """
    x, y = v[0], v[1]
    
    # Valor de la función
    f_val = 100.0 * (y - x**2)**2 + (1.0 - x)**2
    
    # Gradiente
    grad = np.array([
        -400.0 * x * (y - x**2) - 2.0 * (1.0 - x),
        200.0 * (y - x**2)
    ])
    
    # Matriz Hessiana
    hess = np.array([
        [1200.0 * x**2 - 400.0 * y + 2.0, -400.0 * x],
        [-400.0 * x, 200.0]
    ])
    
    return f_val, grad, hess


# 1. Parámetros iniciales requeridos por el problema
x0 = np.array([-1.2, 1.0])
epsilon = 1e-6
max_iter = 5000

# 2. Invocación directa a las 3 funciones del Ejercicio 9
x_cauchy, iters_cauchy, hist_cauchy = min_cauchy(rosenbrock, x0, epsilon, max_iter, regla_armijo)
x_newton, iters_newton, hist_newton = min_newton(rosenbrock, x0, epsilon, max_iter, regla_armijo)
x_bfgs, iters_bfgs, hist_bfgs = min_bfgs(rosenbrock, x0, epsilon, max_iter, regla_armijo)

# =============================================================================
# GRAFICACIÓN DE LAS TASAS DE CONVERGENCIA EN ESCALA LOGARÍTMICA
# =============================================================================

sol_exacta = np.array([1.0, 1.0])

# Métrica de error por iteración: ||x_k - x*||_2
err_cauchy = [np.linalg.norm(x - sol_exacta, 2) for x in hist_cauchy]
err_newton = [np.linalg.norm(x - sol_exacta, 2) for x in hist_newton]
err_bfgs = [np.linalg.norm(x - sol_exacta, 2) for x in hist_bfgs]

plt.figure(figsize=(9, 6))
plt.semilogy(err_cauchy, label=f'Cauchy ({iters_cauchy} iters)', color='red', linewidth=1.5)
plt.semilogy(err_newton, label=f'Newton ({iters_newton} iters)', color='blue', linewidth=2)
plt.semilogy(err_bfgs, label=f'BFGS ({iters_bfgs} iters)', color='green', linewidth=2)

plt.xlabel('Número de Iteraciones ($k$)', fontsize=12)
plt.ylabel('Error $\|x_k - x^*\|_2$ (Escala Logarítmica)', fontsize=12)
plt.title('Tasa de Convergencia en la Función de Rosenbrock', fontsize=13)
plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.show()