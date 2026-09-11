import matplotlib.pyplot as plt
import numpy as np

# Cargar archivos de texto

import matplotlib.pyplot as plt
import numpy as np

# Cargar archivos de texto
import matplotlib.pyplot as plt
import numpy as np

# 1. Definir las rutas de los archivos (funcionan si abres la carpeta 'guia_3' en VS Code)
ruta_archivo_A = 'A_dataset.txt'
ruta_archivo_b = 'b_dataset.txt'

# 2. Cargar los datos usando NumPy
# (Usa 'loadtxt' para archivos de texto con números separados por espacios o tabulaciones)
datos_A = np.loadtxt(ruta_archivo_A)
datos_b = np.loadtxt(ruta_archivo_b)

# --- Tu código para graficar o procesar con matplotlib va aquí abajo ---
print("Archivos cargados con éxito.")
print("Forma de A:", datos_A.shape)
print("Forma de b:", datos_b.shape)



A = np.loadtxt(ruta_archivo_A)
b = np.loadtxt(ruta_archivo_b)

# Solución exacta del sistema original
x = np.linalg.solve(A, b)

delta_x = []
delta_A = []

# 2. Bucle para beta = 1, ..., 10
for beta in range(1, 11):
    eps = 10 ** (-beta)

    # Matriz aleatoria E con la misma forma que A
    E = np.random.randn(*A.shape)

    # Matriz perturbada
    A_tilde = A + eps * E

    # Resolver el sistema perturbado
    x_tilde = np.linalg.solve(A_tilde, b)

    # Cálculo de errores relativos usando norma 2
    err_x = np.linalg.norm(x_tilde - x, 2) / np.linalg.norm(x, 2)
    err_A = np.linalg.norm(A_tilde - A, 2) / np.linalg.norm(A, 2)

    delta_x.append(err_x)
    delta_A.append(err_A)

# 3. Gráfico log-log de delta_x vs delta_A
plt.figure(figsize=(8, 6))
plt.loglog(delta_A, delta_x, "o-", label=r"Error relativo $\delta_x$ vs $\delta_A$")
plt.xlabel(r"Error relativo de la matriz $\delta_A$ (escala log)")
plt.ylabel(r"Error relativo de la solución $\delta_x$ (escala log)")
plt.title("Sensibilidad del Sistema Lineal ante Perturbaciones en A")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend()
plt.show()

# Imprimir el número de condición para la justificación
print("Número de condición kappa_2(A):", np.linalg.cond(A, 2))
