import numpy as np
# Importamos matplotlib.pyplot que es la librería específica para hacer los gráficos
import matplotlib.pyplot as plt
from P1_ej9 import sol_defpos

# EJERCICIO 11: CONDUCCIÓN DE CALOR ESTACIONARIA EN UNA BARRA

# PASO 1: Definición de las variables físicas y de discretización
# El problema nos pide modelar una barra de longitud L = 1 con N = 100 nodos internos.
L = 1.0          # Longitud total de la barra (parámetro físico).
N = 100          # Número de puntos interiores donde calcularemos la temperatura[cite: 1].

# Calculamos el paso espacial 'h' (la distancia fija que hay entre cada nodo)[cite: 1].
# Como la barra mide L y se divide en N+1 subintervalos, cada pedacito mide exactamente h[cite: 1].
h = L / (N + 1)  # En este caso, h = 1 / 101 ≈ 0.0099[cite: 1].

# PASO 2: Creación de la grilla espacial (Coordenadas x_i)
# Guardamos la posición exacta en el eje X de cada uno de los 100 nodos internos[cite: 1].
# El primer nodo empieza en 'h' y el último termina justo antes de L, en 'L - h'[cite: 1].
x = np.linspace(h, L - h, N)

# PASO 3: Construcción de la matriz del sistema (Matriz A)
# El esquema de diferencias finitas centradas para -u''(x) genera una matriz 
# tridiagonal, simétrica y definida positiva de tamaño N x N (100x100)[cite: 1].
# Inicializamos la matriz llena de ceros absolutos usando NumPy.
A = np.zeros((N, N))

# Recorremos fila por fila mediante un bucle para ir rellenando las diagonales.
for i in range(N):
    
    # 1. Diagonal Principal: Cada nodo se relaciona consigo mismo con un peso de 2 / h^2[cite: 1].
    A[i, i] = 2 / h**2          
    
    # 2. Diagonal Inferior (a la izquierda de la diagonal principal):
    # Relaciona al nodo 'i' con el vecino izquierdo 'i-1'[cite: 1]. 
    # Solo se aplica si no estamos en la primera fila (i > 0).
    if i > 0:
        A[i, i-1] = -1 / h**2   
        
    # 3. Diagonal Superior (a la derecha de la diagonal principal):
    # Relaciona al nodo 'i' con el vecino derecho 'i+1'[cite: 1].
    # Solo se aplica si no estamos en la última fila (i < N - 1).
    if i < N - 1:
        A[i, i+1] = -1 / h**2   

# PASO 4: Construcción del vector de carga o fuente (Vector b)
# El término fuente que nos da el ejercicio es f(x) = pi^2 * sin(pi * x)[cite: 1].
# Evaluamos la función de forma directa en cada una de las posiciones 'x' del PASO 2[cite: 1].
# Esto nos genera el vector 'b' de 100 elementos que representa el calor aplicado[cite: 1].
b = (np.pi**2) * np.sin(np.pi * x)

# PASO 5: Resolución del Sistema Lineal (A * u = b)
# Llamamos a tu función 'sol_defpos' desarrollada en el Ejercicio 9[cite: 1].
# Internamente, esta función calcula Cholesky(A) y aplica las sustituciones[cite: 1].
# Nos devuelve el vector 'u' con las 100 temperaturas de los nodos internos.
u = sol_defpos(A, b)


# PASO 6: Reincorporar las condiciones de contorno (u(0) = 0 y u(L) = 0)
# Como el sistema solo calculó los puntos internos, los extremos de la barra no están en 'u'[cite: 1].
# Por ley física sabemos que en x=0 la temperatura es 0, y en x=L también es 0[cite: 1].
# Usamos np.concatenate para "pegar" esos ceros en los extremos y que el gráfico no quede flotando[cite: 1].
x_completo = np.concatenate(([0], x, [L]))     # Eje X completo de 102 puntos (desde 0 hasta 1)[cite: 1].
u_numerico = np.concatenate(([0], u, [0]))     # Eje Y completo de 102 temperaturas (añade 0 en las puntas)[cite: 1].

# PASO 7: Solución Analítica (Exacta) para comparar
# La solución teórica perfecta para esta ecuación diferencial es u(x) = sin(pi * x)[cite: 1].
# La calculamos sobre toda la barra para verificar qué tan preciso fue nuestro método numérico.
u_exacta = np.sin(np.pi * x_completo)

#PASO 8: Visualización gráfica (EXTRA)
# 1. Creamos el lienzo del gráfico con un tamaño de 10 pulgadas de ancho por 6 de alto.
plt.figure(figsize=(10, 6))

# 2. Graficamos la aproximación numérica resuelta con tu Cholesky.
# 'bo' significa: puntos (o) de color azul (b - blue).
plt.plot(x_completo, u_numerico, 'bo', label='Aproximación Numérica (Cholesky)')

# 3. Graficamos la solución analítica exacta sobre el mismo gráfico.
# 'r-' significa: línea continua (-) de color rojo (r - red).
plt.plot(x_completo, u_exacta, 'r-', label='Solución Analítica exacta $u(x) = \\sin(\\pi x)$')

# 4. Agregamos los textos informativos, títulos y etiquetas a los ejes.
plt.title('Perfil de Temperatura Estacionaria en la Barra (Diferencias Finitas)')
plt.xlabel('Posición a lo largo de la barra (x)')
plt.ylabel('Temperatura (u)')

# 5. Activamos una cuadrícula de fondo atenuada para facilitar la lectura de los valores.
plt.grid(True, linestyle='--', alpha=0.5)

# 6. Mostramos el cuadro de referencias (la leyenda) con las etiquetas que definimos en los 'label'.
plt.legend()

# 7. Ordenamos a matplotlib que renderice y despliegue el gráfico final en la pantalla.
plt.show()

