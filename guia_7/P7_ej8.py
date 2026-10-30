import numpy as np

def sol_gradcon(A, b, x, tol, mit):
  A = np.array(A, dtype=float)
  n = A.shape[0] # es la dimension de A y de x
  x = np.array(x, dtype=float)
  b = np.array(b, dtype=float)
  r = b - A @ x
  rho = np.linalg.norm(r, ord = 2)
  d = r
  d_trans = np.transpose(d)

  for k in range(mit):
    if rho <= tol:
       x_mas = x
    else:
      v = A @ d
      t = (rho**2) / (d_trans @ v)
      x = x + t * d
      r = r - t * v
      rho_mas = np.linalg.norm(r, ord = 2)
      s = (rho_mas**2) / (rho**2)
      d = r + s * d
      rho = rho_mas
      x_mas = x
  return x_mas