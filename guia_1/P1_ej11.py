import numpy as np
# Importamos matplotlib.pyplot que es la librería específica para hacer los gráficos
import matplotlib.pyplot as plt
from P1_ej9 import sol_defpos

# EJERCICIO 11: CONDUCCIÓN DE CALOR ESTACIONARIA EN UNA BARRA

# PASO 1: Definición de las variables físicas y de discretización
# El problema nos pide modelar una barra de longitud L = 1 con N = 100 nodos internos.
L = 1.0          # Longitud total de la barra (parámetro físico).
N = 100          # Número de puntos interiores donde calcularemos la temperatura.

# Calculamos el paso espacial 'h' (la distancia fija que hay entre cada nodo).
# Como la barra mide L y se divide en N+1 subintervalos, cada pedacito mide exactamente h.
h = L / (N + 1)  # En este caso, h = 1 / 101 ≈ 0.0099.

# PASO 2: Creación de la grilla espacial (Coordenadas x_i)
# Guardamos la posición exacta en el eje X de cada uno de los 100 nodos internos.
# El primer nodo empieza en 'h' y el último termina justo antes de L, en 'L - h'.
x = np.linspace(h, L - h, N)

# PASO 3: Construcción de la matriz del sistema (Matriz A)
# El esquema de diferencias finitas centradas para -u''(x) genera una matriz 
# tridiagonal, simétrica y definida positiva de tamaño N x N (100x100).
# Inicializamos la matriz llena de ceros absolutos usando NumPy.
A = np.zeros((N, N))

# Recorremos fila por fila mediante un bucle para ir rellenando las diagonales.
for i in range(N):
    
    # 1. Diagonal Principal: Cada nodo se relaciona consigo mismo con un peso de 2 / h^2.
    A[i, i] = 2 / h**2          
    
    # 2. Diagonal Inferior (a la izquierda de la diagonal principal):
    # Relaciona al nodo 'i' con el vecino izquierdo 'i-1'. 
    # Solo se aplica si no estamos en la primera fila (i > 0).
    if i > 0:
        A[i, i-1] = -1 / h**2   
        
    # 3. Diagonal Superior (a la derecha de la diagonal principal):
    # Relaciona al nodo 'i' con el vecino derecho 'i+1'.
    # Solo se aplica si no estamos en la última fila (i < N - 1).
    if i < N - 1:
        A[i, i+1] = -1 / h**2   

# PASO 4: Construcción del vector de carga o fuente (Vector b)
# El término fuente que nos da el ejercicio es f(x) = pi^2 * sin(pi * x).
# Evaluamos la función de forma directa en cada una de las posiciones 'x' del PASO 2.
# Esto nos genera el vector 'b' de 100 elementos que representa el calor aplicado.
b = (np.pi**2) * np.sin(np.pi * x)

# PASO 5: Resolución del Sistema Lineal (A * u = b)
# Llamamos a tu función 'sol_defpos' desarrollada en el Ejercicio 9.
# Internamente, esta función calcula Cholesky(A) y aplica las sustituciones.
# Nos devuelve el vector 'u' con las 100 temperaturas de los nodos internos.
u = sol_defpos(A, b)


# PASO 6: Reincorporar las condiciones de contorno (u(0) = 0 y u(L) = 0)
# Como el sistema solo calculó los puntos internos, los extremos de la barra no están en 'u'.
# Por ley física sabemos que en x=0 la temperatura es 0, y en x=L también es 0.
# Usamos np.concatenate para "pegar" esos ceros en los extremos y que el gráfico no quede flotando.
x_completo = np.concatenate(([0], x, [L]))     # Eje X completo de 102 puntos (desde 0 hasta 1).
u_numerico = np.concatenate(([0], u, [0]))     # Eje Y completo de 102 temperaturas (añade 0 en las puntas).

# PASO 7: Solución Analítica (Exacta) para comparar
# La solución teórica perfecta para esta ecuación diferencial es u(x) = sin(pi * x).
# La calculamos sobre toda la barra para verificar qué tan preciso fue nuestro método numérico.
u_exacta = np.sin(np.pi * x_completo)

#PASO 8: Visualización gráfica (EXTRA)
# 1. Creamos el plano del gráfico con un tamaño de 10 cm de ancho por 6 de alto.
plt.figure(figsize=(10, 6))

# 2. Graficamos la aproximación numérica resuelta con tu Cholesky.
# 'bo' significa: puntos (o) de color azul (b - blue).
plt.plot(x_completo, u_numerico, 'bo', label='Aproximación Numérica (Cholesky)')

# 3. Graficamos la solución analítica exacta sobre el mismo gráfico.
# 'r-' significa: línea continua (-) de color rojo (r - red).
plt.plot(x_completo, u_exacta, 'r-', label='Solución Analítica exacta u(x) = \\sin(\\pi x)$')

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


"""
EXPLICAION PASO 3:
Es un despeje matemático. Lo que estás haciendo en el Paso 3 es agarrar la ecuación 
diferencial que me dan, despejarla de forma que te queden todas las incógnitas ordenadas
de izquierda a derecha, y meter esos coeficientes (los números que multiplican a las u)
en los casilleros correspondientes de la matriz A.  
Es deir: El Despeje: Al meter la aproximación en -u''(x_i) = f(x_i), el signo menos 
de la ecuación te cambia los signos de todo lo de adentro. Al limpiarla, 
te queda ordenada nodo por nodo:
(-1/h^2)*u_(i-1) + (2/h^2)*u_i + (-1/h^2)u_(i+1) = f(x_i)
La Matriz: Como tenés 100 ecuaciones acopladas (una para cada punto de la barra), usás la matriz A
como una "tabla organizadora".  La columna del medio (i) guarda el valor del nodo actual (2/h^2$).
La columna izquierda (i-1) guarda el del vecino izquierdo (-1/h^2).  
La columna derecha (i+1) guarda el del vecino derecho (-1/h^2). 
"""
