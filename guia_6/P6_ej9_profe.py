#PROFE:
import sys
import os

# 1. Definimos la ruta exacta de tu carpeta guia_4
# (Asegúrate de respetar las mayúsculas y minúsculas exactas del sistema)
ruta_guia4 = "/home/usuario_kmom/analisis_numerico2/guia_4"

# 2. Le avisamos a Python que busque en esa carpeta
if ruta_guia4 not in sys.path:
    sys.path.append(ruta_guia4)

# 3. Ahora ya puedes importar sin problemas
from P4_ej3_profe import house


def fhess(A, met):
    n = A.shape[0]
    H = A.copy()
    Q  = np.eye(n)

    if met == 'house':
        for j in range(n-2):
            rho, u = house(H[j+1:, j])
            w  = rho*u
            v = u.T@H[j+1: , j:]
            H[j+1: , j:] = H[j+1: , j:]-np.outer(w, v)
            H[: , j+1:] = H[: , j+1:]-H[: , j+1:]@np.outer(w,u)
            Q[: , j+1:] = Q[: , j+1:]-Q[: , j+1:]@np.outer(w,u)
    
    # Debo givens 
    return H, Q

A  = np.random.random((4,4))

H, Q = fhess(A, 'house')

print(H)