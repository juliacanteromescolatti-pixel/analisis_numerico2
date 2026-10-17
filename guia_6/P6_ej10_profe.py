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
from P4_ej3_profe import rot_givens, house
from P6_ej9 import fhess

def autqr_hess(A, m):
    
    H, Q = fhess(A)
    n = H.shape[0]
    for k in range(m):
        for j in range(n-1):
            c,s = rot_givens(H[j,j],H[j+1,j])
            G = np.array([[c, -s],
                          [s, c]])
            H[[j ,j+1], j:] = G@H[[j ,j+1], j:] 
#??????