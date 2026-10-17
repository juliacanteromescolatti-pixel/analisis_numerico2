#===============================================================================
# PROFE
#===============================================================================

import numpy as np

import sys
sys.path.append('..')
from practico2.ejercicio_10 import egaussp
from practico2.ejercicio_11 import sol_egauss

# Version norma infinito
def autpotencias_inf(A, q_0, err=1e-10, m=500):
    q_tilde = A @ q_0
    rho_tilde = np.inf
    for k in range(m):
        j = np.argmax(np.abs(q_tilde))
        q = q_tilde/q_tilde[j]
        q_tilde = A @ q
        rho = q_tilde[j]

        if np.abs(rho - rho_tilde) < err:
            print("iteraciones:", k)
            return q, rho