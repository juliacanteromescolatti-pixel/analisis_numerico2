import numpy as np

# 1. Definición de los datos entregados en el práctico
fecundidad = np.array([0.0, 0.2, 0.9, 0.9, 0.9, 0.8, 0.3])
supervivencia = np.array([0.3, 0.7, 0.9, 0.9, 0.9, 0.6, 0.0])
n0 = np.array([10, 2, 8, 5, 12, 0, 1], dtype=float)

p = len(fecundidad)

# 2. Construcción de la Matriz de Leslie (L)
L = np.zeros((p, p))

# Primera fila: Tasas de fecundidad (f_i)
L[0, :] = fecundidad

# Subdiagonal principal: Tasas de supervivencia (s_i)
# s_i es la tasa con la que la faja i pasa a la faja i+1
for i in range(p - 1):
    L[i + 1, i] = supervivencia[i]

print("Matriz de Leslie (L):")
print(np.round(L, 2))
print("-" * 50)

# 3. Cálculo de autovalores y autovectores
autovalores, autovectores = np.linalg.eig(L)

# Identificamos el autovalor dominante (el de mayor módulo/módulo real)
idx_dominante = np.argmax(np.abs(autovalores))
lambda_1 = np.real(autovalores[idx_dominante])
autovector_asociado = np.real(autovectores[:, idx_dominante])

# Normalizamos el autovector para facilitar su interpretación (suma de componentes = 1)
autovector_normalizado = autovector_asociado / np.sum(autovector_asociado)

print(f"Autovalor dominante (λ1): {lambda_1:.4f}")
print("-" * 50)

# 4. Análisis del comportamiento poblacional
print("DETERMINACIÓN DEL COMPORTAMIENTO POBLACIONAL:")
if lambda_1 > 1:
    print(f"Dado que λ1 = {lambda_1:.4f} > 1, la población CRECE EXPONENCIALMENTE.")
elif lambda_1 < 1:
    print(f"Dado que λ1 = {lambda_1:.4f} < 1, la población DECRECE EXPONENCIALMENTE hacia la extinción.")
else:
    print(f"Dado que λ1 = {lambda_1:.4f} = 1, la POBLACIÓN ES ESTABLE.")

print("-" * 50)
print("Distribución etaria límite (Autovector asociado normalizado):")
for i, dist in enumerate(autovector_normalizado):
    print(f" Faja etaria {i+1}: {dist * 100:.2f}%")

# 5. Simulación del avance en el tiempo (opcional para verificar)
num_anios = 50
n_t = n0.copy()
for t in range(num_anios):
    n_t = L @ n_t

print("-" * 50)
print(f"Población total inicial: {np.sum(n0):.0f}")
print(f"Población total proyectada tras {num_anios} años: {np.sum(n_t):.2f}")