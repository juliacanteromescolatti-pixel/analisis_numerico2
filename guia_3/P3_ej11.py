import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Obtener la ruta del directorio donde está guardado este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Definir rutas absolutas para evitar el FileNotFoundError
ruta_archivo_A = os.path.join(script_dir, "A_dataset.txt")
ruta_archivo_b = os.path.join(script_dir, "b_dataset.txt")

# 3. Cargar los datos usando NumPy
A = np.loadtxt(ruta_archivo_A)
b = np.loadtxt(ruta_archivo_b)

print("Archivos cargados con éxito.")
print("Forma de A:", A.shape)
print("Forma de b:", b.shape)

# Solución exacta del sistema original
x = np.linalg.solve(A, b)

delta_x = []
delta_A = []

# 4. Bucle para beta = 1, ..., 10
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

# 5. Gráfico log-log de delta_x vs delta_A
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