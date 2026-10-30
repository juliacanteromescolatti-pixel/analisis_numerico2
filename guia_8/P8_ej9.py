import numpy as np

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