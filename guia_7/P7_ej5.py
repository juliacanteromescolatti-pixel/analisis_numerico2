import numpy as np

def sol_cauchy(A, b, x, tol, mit):
  A = np.array(A, dtype=float)
  n = A.shape[0] # es la dimension de A y de x
  x = np.array(x, dtype=float)
  b = np.array(b, dtype=float)
  r = b - A @ x
  r_trans = np.transpose(r)
  rho = np.linalg.norm(r, ord = 2)
  for k in range(mit):
    if rho <= tol:
       x_mas = x
    else:
      v = A @ r
      t = (rho**2) / (r_trans @ v)
      x = x + t * r
      r = r - t * v
      rho = np.linalg.norm(r, ord = 2)
      x_mas = x
  return x_mas
