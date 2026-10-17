import numpy as np


# (Opcional) Usamos tu función fhess previa para la reducción a Hessenberg
def fhess(A, met="hholder"):
  A = np.array(A, dtype=float)
  n = A.shape[0]
  Q = np.eye(n)

  if met == "hholder":
    for k in range(n - 2):
      x = A[k + 1 :, k]
      norm_x = np.linalg.norm(x)
      if norm_x == 0:
        continue
      v = x.copy()
      v[0] += np.sign(v[0]) * norm_x
      v = v / np.linalg.norm(v)
      H_sub = np.eye(len(v)) - 2 * np.outer(v, v)
      A[k + 1 :, k:] = np.dot(H_sub, A[k + 1 :, k:])
      A[:, k + 1 :] = np.dot(A[:, k + 1 :], H_sub)
      Q[:, k + 1 :] = np.dot(Q[:, k + 1 :], H_sub)
  else:
    raise ValueError("Método no válido para fhess en este ejemplo.")

  return Q, A  # Retorna Q y H (H está en A)


def autqr_hess(A, m=500, tol=1e-10):
  """Calcula los autovalores de una matriz A usando iteraciones QR

  partiendo de su forma de Hessenberg y utilizando rotaciones de Givens.
  """
  A = np.array(A, dtype=float)
  n = A.shape[0]

  # 1. Reducción inicial a la forma de Hessenberg superior
  _, H = fhess(A, met="hholder")

  # 2. Iteraciones QR
  for iteracion in range(m):
    H_old = H.copy()

    # Descomposición QR de la matriz de Hessenberg usando rotaciones de Givens
    # (Para mantener la estructura de Hessenberg de forma eficiente)
    Q_qr = np.eye(n)
    R = H.copy()

    for i in range(n - 1):
      a = R[i, i]
      b = R[i + 1, i]
      if b != 0:
        r = np.hypot(a, b)
        c = a / r
        s = -b / r

        # Aplicar rotación de Givens a R por la izquierda (filas i e i+1)
        row1 = R[i, :]
        row2 = R[i + 1, :]
        R[i, :] = c * row1 - s * row2
        R[i + 1, :] = s * row1 + c * row2

        # Acumular la matriz ortogonal Q_qr
        col1_q = Q_qr[:, i]
        col2_q = Q_qr[:, i + 1]
        Q_qr[:, i] = c * col1_q - s * col2_q
        Q_qr[:, i + 1] = s * col1_q + c * col2_q

    # Siguiente iteración: H_{k+1} = R * Q_qr
    H = np.dot(R, Q_qr)

    # Criterio de convergencia (si los elementos debajo de la diagonal son casi cero o H no cambia)
    if np.linalg.norm(H - H_old, ord=np.inf) < tol:
      break

  # Los autovalores se encuentran en la diagonal principal
  autovalores = np.diagonal(H)
  return autovalores


# --- TESTEANDO CON LA MATRIZ DEL EJERCICIO ---

# Definimos la matriz B (3x3)
B = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)

# Identidad de 3x3 y matriz nula de 3x3
I = np.eye(3)
Z = np.zeros((3, 3))

# Construcción de la matriz A por bloques (9x9)
A1 = np.hstack([B, -I, Z])
A2 = np.hstack([-I, B, -I])
A3 = np.hstack([Z, -I, B])
A = np.vstack([A1, A2, A3])

# Ejecutamos nuestra función
autovalores_obtenidos = autqr_hess(A, m=500)

print("Autovalores calculados con autqr_hess:")
print(np.sort(autovalores_obtenidos))

# Comprobación con numpy.linalg.eig para verificar
autovalores_ref = np.linalg.eigvalsh(A)
print("\nAutovalores de referencia (numpy):")
print(np.sort(autovalores_ref))

#===============================================================================
