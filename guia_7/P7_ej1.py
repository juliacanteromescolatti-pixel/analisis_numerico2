
import numpy as np
import matplotlib.pyplot as plt

# Definición directa del sistema Ax = b
A = np.array([[10.0, 1.0],
              [4.0,  6.0]])
b = np.array([3.0, 9.0])

# Vector inicial x0
x0 = np.array([1.0, 1.0])

# --- Iteración 1 ---
# x1 = (3 - y0)/10 ,  y1 = (9 - 4*x0)/6
x1 = np.zeros(2)
x1[0] = (b[0] - A[0, 1] * x0[1]) / A[0, 0]
x1[1] = (b[1] - A[1, 0] * x0[0]) / A[1, 1]

# --- Iteración 2 ---
# x2 = (3 - y1)/10 ,  y2 = (9 - 4*x1)/6
x2 = np.zeros(2)
x2[0] = (b[0] - A[0, 1] * x1[1]) / A[0, 0]
x2[1] = (b[1] - A[1, 0] * x1[0]) / A[1, 1]

# Solución exacta
x_exacta = np.linalg.solve(A, b)

# Mostrar valores
print(f"x0: {x0}")
print(f"x1: {x1}")
print(f"x2: {x2}")
print(f"Solución exacta x*: {x_exacta}")

# --- Gráfico de los puntos ---
puntos = np.array([x0, x1, x2])

plt.figure(figsize=(8, 6))

# Trazar iteraciones de Jacobi
plt.plot(puntos[:, 0], puntos[:, 1], 'o--', color='blue', label='Iteraciones de Jacobi')

# Trazar solución exacta
plt.scatter(x_exacta[0], x_exacta[1], color='red', s=100, zorder=5, label='Solución exacta ($x^*$)')

# Etiquetas en el gráfico
plt.annotate('$x_0 (1, 1)$', (x0[0], x0[1]), textcoords="offset points", xytext=(10, 5))
plt.annotate(f'$x_1 ({x1[0]:.2f}, {x1[1]:.2f})$', (x1[0], x1[1]), textcoords="offset points", xytext=(10, 5))
plt.annotate(f'$x_2 ({x2[0]:.2f}, {x2[1]:.2f})$', (x2[0], x2[1]), textcoords="offset points", xytext=(10, -15))
plt.annotate(f'$x^*$ ({x_exacta[0]:.2f}, {x_exacta[1]:.2f})', (x_exacta[0], x_exacta[1]), textcoords="offset points", xytext=(-60, -15), color='red')

plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Ejercicio 1: Iteraciones del Método de Jacobi')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()