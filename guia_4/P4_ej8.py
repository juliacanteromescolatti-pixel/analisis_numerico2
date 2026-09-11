import numpy as np
#Importamos todas las gunciones necesarias:
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 1º PRIMERO: Le enseñamos a Python dónde están tus carpetas
sys.path.append('/home/kmom/analisis_numerico2')
sys.path.append('/home/kmom/analisis_numerico2/guia_2')
sys.path.append('/home/kmom/analisis_numerico2/guia_1')

# 2º SEGUNDO: Recién ahora hacemos la importación
from guia_2.P2_ej11 import sol_egauss
from guia_1.P1_ej5 import cholesky
from P4_ej7 import sol_cuadmin          

# =============================================================================
# EJERCICIO 8 — A(eps) x = [1,1,1]^T cuando eps -> 0, comparando:
#   (a) ecuación normal resuelta con Cholesky (ejercicio 5 de la guia 1)
#   (a') ecuación normal resuelta con LU/Gauss con pivoteo (sol_egauss, ejercicio 11 de la guia 2)
#   (b) descomposición QR (sol_cuadmin, práctico 4 ejercicio 7)
# =============================================================================
#
# RECORDATORIO DE POR QUÉ HACEMOS ESTA COMPARACIÓN
# El x que minimiza ||A x - b||^2 es el mismo, lo mires como lo mires (eso
# lo demostraste en el ejercicio 6). La diferencia entre estos métodos es
# puramente NUMÉRICA: qué tan bien se comporta cada uno en punto flotante
# cuando el sistema está mal condicionado.
#
# - Ecuación normal (Cholesky o LU sobre A^T A x = A^T b):
#       cond(A^T A) = cond(A)^2
#   Elevar al cuadrado un número de condición ya grande lo manda mucho más
#   rápido al límite de precisión de la máquina (~10^16 en double).
#   Cholesky además exige que A^T A sea definida positiva: si por errores
#   de redondeo deja de "verse" definida positiva, sol_cholesky va a
#   lanzar el ValueError que programamos en P4_ej8_cholesky.py.
#   LU con pivoteo parcial (sol_egauss) no tiene ese chequeo, así que
#   puede devolver un resultado sin sentido (o con división por un pivote
#   casi nulo) sin avisar.
#
# - QR (sol_cuadmin): nunca arma A^T A. El error que introduce es
#   proporcional a cond(A), no a cond(A)^2, así que en teoría aguanta
#   sistemas mucho peor condicionados antes de romperse.
# =============================================================================


def A_eps(eps):
    """Matriz A(eps) = [[1,1],[eps,0],[0,eps]] del enunciado."""
    return np.array([[1.0, 1.0],
                      [eps, 0.0],
                      [0.0, eps]])


def solucion_exacta(eps):
    """
    Solución analítica de min ||A(eps) x - b||^2 con b=[1,1,1]^T.
    Por simetría del problema, x1=x2=x, y de la ecuación normal a mano:
        (1+eps^2) x + x = 1+eps   =>   x = (1+eps)/(2+eps^2)
    Se usa solo como referencia para medir el error de cada método.
    """
    x = (1.0 + eps) / (2.0 + eps**2)
    return np.array([x, x])


if __name__ == "__main__":
    b = np.array([1.0, 1.0, 1.0])
    epsilons = [1e-1, 1e-3, 1e-5, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-14, 1e-16]

    print(f"{'eps':>8} | {'cond(A)':>10} | {'cond(AtA)':>10} | "f"{'err_chol':>10} | {'err_LU(P2)':>10} | {'err_qr':>10}")

    for eps in epsilons:
        A = A_eps(eps)
        AtA = A.T @ A          # matriz de la ecuación normal, (2,2)
        Atb = A.T @ b          # lado derecho de la ecuación normal
        x_exacta = solucion_exacta(eps)

        cond_A = np.linalg.cond(A)
        cond_AtA = np.linalg.cond(AtA)

        #(a) Ecuación normal resuelta con Cholesky 
        # AtA x = Atb  ,  AtA = L L^T
        # Si el determinante es casi cero o negativo, Cholesky no se puede calcular
        if np.linalg.det(AtA) <= 1e-15:
            err_chol = np.nan
        else:
            # Usamos la Cholesky estándar pasándole un solo dato como corresponde
            L_numpy = np.linalg.cholesky(AtA)
            x_chol = np.linalg.solve(L_numpy.T, np.linalg.solve(L_numpy, Atb))
            err_chol = np.max(np.abs(x_chol - x_exacta))

        #(a') Ecuación normal resuelta con LU + pivoteo (práctico 2) 
        x_lu = sol_egauss(AtA, Atb)
        
        # Si tu función da "None" porque la matriz es singular, le asignamos NaN
        if x_lu is None:
            err_lu = np.nan
        else:
            err_lu = np.max(np.abs(x_lu - x_exacta))

        #(b) QR (práctico 4, ejercicio 7) 
        x_qr = sol_cuadmin(A, b)
        err_qr = np.max(np.abs(x_qr - x_exacta))

        print(f"{eps:8.0e} | {cond_A:10.2e} | {cond_AtA:10.2e} | "f"{err_chol:10.2e} | {err_lu:10.2e} | {err_qr:10.2e}")

#np.nan significa "No es un Número", es decir pude ser 0/0 o infinito/infinito.