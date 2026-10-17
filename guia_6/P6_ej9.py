"""Calcula la forma de Hessenberg superior H = Q^T * A * Q
  Parámetros:
  - A: matriz cuadrada de tamaño n x n (numpy array o lista de listas).
  - met: string ('hholder' para Householder, 'givens' para Givens).

  Retorna:
  - Q: matriz ortogonal.
  - H: matriz en forma de Hessenberg superior."""


import numpy as np

def fhess(A, met="hholder"):
  A = np.array(A, dtype=float)
  n = A.shape[0]
  Q = np.eye(n)
  if met == "hholder":
    for k in range(n - 2):
      # Vector x a anular en la columna k (desde la fila k+1 hasta el final)
      x = A[k + 1 :, k]
      norm_x = np.linalg.norm(x)

      if norm_x == 0:
        continue
      # Construcción del vector v de Householder
      v = x.copy()
      v[0] += np.sign(v[0]) * norm_x
      v = v / np.linalg.norm(v)

      # Matriz de reflexión local
      H_sub = np.eye(len(v)) - 2 * np.outer(v, v)

      # Aplicar transformación por la izquierda: A = (I - v)(A)
      A[k + 1 :, k:] = np.dot(H_sub, A[k + 1 :, k:])
      # Aplicar transformación por la derecha: A = (A)(I - v)
      A[:, k + 1 :] = np.dot(A[:, k + 1 :], H_sub)

      # Acumular la matriz ortogonal Q
      Q[:, k + 1 :] = np.dot(Q[:, k + 1 :], H_sub)

  elif met == "givens":
    for k in range(n - 2):
      for i in range(n - 1, k + 1, -1):
        # Queremos anular el elemento A[i, k] usando A[i-1, k]
        a = A[i - 1, k]
        b = A[i, k]

        if b != 0:
          r = np.hypot(a, b)
          c = a / r
          s = -b / r

          # Aplicar rotación por la izquierda a las filas i-1 e i
          row1 = A[i - 1, :]
          row2 = A[i, :]
          A[i - 1, :] = c * row1 - s * row2
          A[i, :] = s * row1 + c * row2

          # Aplicar rotación por la derecha a las columnas i-1 e i
          col1 = A[:, i - 1]
          col2 = A[:, i]
          A[:, i - 1] = c * col1 - s * col2
          A[:, i] = s * col1 + c * col2

          # Acumular la matriz ortogonal Q
          col1_q = Q[:, i - 1]
          col2_q = Q[:, i]
          Q[:, i - 1] = c * col1_q - s * col2_q
          Q[:, i] = s * col1_q + c * col2_q
  else:
    raise ValueError("Método no válido. Usa 'hholder' para Householder o 'givens' para Givens.")
  # H es la matriz A modificada a su forma de Hessenberg
  H = A
  return Q, H


# Matriz de ejemplo de 4x4
A = [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 1], [2, 3, 1, 5]]

# 1. Usando el método de Householder (por defecto o especificándolo)
Q_h, H_h = fhess(A, met="hholder")
print("--- Método Householder ---")
print("Matriz H (Hessenberg)", H_h)
print("Matriz Q ", Q_h)

# 2. Usando el método de Givens
Q_g, H_g = fhess(A, met="givens")
print("\n--- Método Givens ---")
print("Matriz H (Hessenberg) ",H_g)
print("Matriz Q ",Q_g)

#===============================================================================
