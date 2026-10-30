#=======================================================================================
#EJERCICIO 4:
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Función que retorna F(x) y la Matriz Jacobiana J(x)
# -------------------------------------------------------------------------
def func_ejercicio4(x):
    x1, x2 = x[0], x[1]
    
    # Vector de residuos F(x)
    F = np.array([
        x1**2 - x1 - x2,
        (x1**2) / 16.0 + x2**2 - 1.0
    ])
    
    # Matriz Jacobiana J(x)
    J = np.array([
        [2.0 * x1 - 1.0, -1.0],
        [x1 / 8.0, 2.0 * x2]
    ])
    
    return F, J

# -------------------------------------------------------------------------
# 2. Algoritmo de Newton Exacto (usando la función sol_newton, EJERICIO 3)
# -------------------------------------------------------------------------
def sol_newton(func, x0, tol=1e-6, m=100):
    x = np.array(x0, dtype=float)
    for k in range(m):
        F, J = func(x)
        if np.linalg.norm(F, 2) < tol:
            return x, k
        d = np.linalg.solve(J, -F)
        x = x + d
    return x, m

# -------------------------------------------------------------------------
# 3. Búsqueda de todos los puntos de intersección
#    (Probamos con distintos puntos iniciales x0)
# -------------------------------------------------------------------------
puntos_iniciales = [
    [-3.0, 0.5],   # x0 para intersección cuadrante superior izquierdo
    [-1.0, -0.8],  # x0 para intersección cuadrante inferior izquierdo
    [1.5, -0.8],   # x0 para intersección cuadrante inferior derecho
    [2.0, 0.8]     # x0 para intersección cuadrante superior derecho
]

soluciones = []
tol = 1e-8
m = 50

for x0 in puntos_iniciales:
    raiz, iters = sol_newton(func_ejercicio4, x0, tol=tol, m=m)
    
    # Evitar duplicados
    es_nuevo = True
    for s in soluciones:
        if np.linalg.norm(s - raiz) < 1e-4:
            es_nuevo = False
            break
            
    if es_nuevo:
        soluciones.append(raiz)

print("Puntos de intersección hallados:")
for i, sol in enumerate(soluciones, 1):
    print(f" Punto {i}: x1 = {sol[0]:.6f}, x2 = {sol[1]:.6f}")

# -------------------------------------------------------------------------
# 4. Visualización Gráfica (Parábola, Elipse y Puntos hallados)
# -------------------------------------------------------------------------
# Malla para graficar las curvas implícitas
x1_vals = np.linspace(-4.5, 4.5, 400)
x2_vals = np.linspace(-1.5, 2.5, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Evaluaciones implícitas
Z_parabola = X1**2 - X1 - X2
Z_elipse = (X1**2) / 16.0 + X2**2 - 1.0

plt.figure(figsize=(9, 6))

# Graficar la Parábola y la Elipse (donde Z == 0)
plt.contour(X1, X2, Z_parabola, levels=[0], colors='blue', linewidths=2)
plt.contour(X1, X2, Z_elipse, levels=[0], colors='red', linewidths=2)

# Crear leyenda manual para las curvas
plt.plot([], [], color='blue', label=r'Parábola: $x_2 = x_1^2 - x_1$')
plt.plot([], [], color='red', label=r'Elipse: $x_1^2/16 + x_2^2 = 1$')

# Graficar los puntos de intersección encontrados con Newton
soluciones = np.array(soluciones)
plt.scatter(soluciones[:, 0], soluciones[:, 1], color='black', zorder=5, s=60, label='Intersecciones (Newton)')

# Etiquetar cada punto
for i, sol in enumerate(soluciones, 1):
    plt.annotate(f'P{i}', (sol[0], sol[1]), textcoords="offset points", xytext=(8,8), ha='left', weight='bold')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title('Intersección entre Parábola y Elipse (Método de Newton)', fontsize=13)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis('equal')
plt.show()