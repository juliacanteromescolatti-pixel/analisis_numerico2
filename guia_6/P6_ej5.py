# INCISO A
# NORMA INFINITO np.linalg.norm(A, ord=np.inf) OTRO FORMA DE PONER ALGOP INDINITO float('inf')
import numpy as np

def autpotenciasinf(A, q, err, mit):
  m, n = A.shape
  q_pirm = A @ q
  ro_prim = np.inf

  for k in range (1, m+1):
    j = np.argmax(np.abs(q_prim))
    q_norm = q_prim/q_prim[j]
    q_prim = A @ q_norm
    ro = q_prim[j]

    if np.abs(ro - ro_prim <= err): # control de convergencia
      break
    else:
      ro_prim = ro

  return ro, q_norm

#==========================================================================================================
# INCISO B
# NORMA 2 np.linalg.norm(A, ord=2)

def autpotencias2(A, q, err, mit):
  m, n = A.shape
  q_pirm = A @ q
  ro_prim = q.T @ q_prim / np.linalg.norm(q, ord = 2)**2

  for k in range (1, m+1):
    q_norm = q_prim/np.linalg.norm(q_prim, ord = 2)
    q_prim = A @ q_norm
    ro = q_norm.T @ q_prim

    if np.abs(ro - ro_prim <= err): # control de convergencia
      break
    else:
      ro_prim = ro

  return ro, q_norm

#===========================================================================================================
# INCISO C

def autrayleigh(A, q0, err, mit):
  m, n = A.shape
  q = q0/np.linalg.norm(q0, ord = 2)
  q_pirm = q
  ro = q.T @ A @ q
  I = np.eye(n)
  for k in range (1, m+1):

        # Resolver (A - rho * I) z = q_hat
        # Nota: (A - rho * I) se vuelve casi singular a medida que rho -> lambda,
        # lo cual es esperado en este método.
        z = np.linalg.solve(A - ro * I, q_prim)

        sigma = np.linalg.norm(z, 2)
        q = z / sigma
        theta = float(q.T @ q_prim / sigma)

        # Criterio de convergencia
        if np.abs(theta) <= err:
            return q, ro + theta

        # Actualización de variables
        q_prim = q.copy()
        ro = ro + theta
    # Paso 2 si se alcanzan las iteraciones máximas
  return q, ro + theta
