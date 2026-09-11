"""
EJERCICIO 9 - Análisis Numérico II
-----------------------------------
Reutiliza:
  - tu función `cholesky` de la Guía 1 (P1_ej5)
  - tu función `sol_cuadmin` del Ejercicio 7 del Práctico 4 (P4_ej7),
    que resuelve mínimos cuadrados vía QR con permutación de columnas.

Objetivo: resolver minimizar_x ||Ax - b||^2 para la A y b del enunciado,
comparando:
    1) Ecuaciones normales (A^T A x = A^T b) resueltas con TU Cholesky
    2) TU función de cuadrados mínimos vía QR

NOTA IMPORTANTE: este script depende de tus archivos locales
(guia_1/P1_ej5.py y P4_ej7.py). No lo pude ejecutar en mi entorno porque
no tengo esos archivos, así que revisá los resultados en tu máquina.
"""

import numpy as np
import sys

# =========================================================
# CONEXIÓN CON TUS MÓDULOS DE GUÍAS ANTERIORES
# =========================================================
# sys.path.append agrega carpetas a la lista de lugares donde Python
# busca módulos para importar. Sin esto, Python no encontraría tus
# archivos porque están en otra carpeta distinta a este script.
sys.path.append('/home/kmom/analisis_numerico2')
sys.path.append('/home/kmom/analisis_numerico2/guia_1')

# Importamos las funciones ya implementadas y testeadas en otras guías,
# en vez de reescribirlas de cero (evita duplicar trabajo y errores).
from guia_1.P1_ej5 import cholesky as mi_cholesky   # factor de Cholesky: A = L L^T
from P4_ej7 import sol_cuadmin as mi_sol_qr          # resuelve min ||Ax-b||^2 vía QR


def ejercicio_9(n):
    # =========================================================
    # 1. CONSTRUCCIÓN DE LA MATRIZ A (tamaño n x (n-2))
    # =========================================================
    # Matriz banda: cada columna j tiene -1, 2, -1 en las filas j, j+1, j+2.
    # Esto reproduce la estructura tridiagonal del enunciado:
    #   primera columna:  -1, 2, -1, 0, 0, ...
    #   última columna:    ..., 0, 0, -1, 2, -1
    A = np.zeros((n, n - 2))          # matriz de ceros, n filas, n-2 columnas
    for j in range(n - 2):            # recorre cada columna (0 a n-3)
        A[j, j] = -1                  # entrada -1 en la fila j
        A[j + 1, j] = 2                # entrada  2 en la fila j+1
        A[j + 2, j] = -1               # entrada -1 en la fila j+2

    # ---------------------------------------------------------
    # BLOQUE "AJUSTE FIEL A LA FOTO" (revisar si hace falta)
    # ---------------------------------------------------------
    # OJO: tal como está escrito, este bloque es redundante.
    # col_central es la columna del medio, y las líneas de abajo
    # vuelven a poner exactamente los mismos valores que el bucle
    # de arriba ya puso para j = col_central:
    #     A[col_central + 1, col_central] = 2   (ya lo hace el for)
    #     A[col_central + 2, col_central] = -1  (ya lo hace el for)
    # Es decir, no cambia nada en la matriz. Si tu fotocopia muestra
    # otro valor (por ejemplo "gamma" distinto de 2, o "gamma-1"
    # distinto de 1) en esa columna central, hay que poner ESE valor
    # acá, no repetir el mismo. Si no hay diferencia real, se puede
    # borrar este bloque entero sin que cambie el resultado.
    col_central = (n - 2) // 2        # índice de la columna central
    if n > 2:
        A[col_central + 1, col_central] = 2.0      # (redundante con el for)
        A[col_central + 2, col_central] = -1.0      # (redundante con el for)

    # =========================================================
    # 2. CONSTRUCCIÓN DEL VECTOR b (tamaño n)
    # =========================================================
    b = np.zeros(n)
    b[0] = 1.0                        # primer elemento = 1
    b[-1] = 1.0                       # último elemento = 1

    # =========================================================
    # 3. MÉTODO 1: ECUACIONES NORMALES + TU CHOLESKY
    # =========================================================
    # La solución de mínimos cuadrados también resuelve el sistema
    # cuadrado (n-2)x(n-2):     A^T A x = A^T b
    ATA = A.T @ A                     # matriz (n-2)x(n-2), simétrica def. positiva
    ATb = A.T @ b                     # lado derecho del sistema normal

    # Chequeo de seguridad: si A^T A fuera (casi) singular, Cholesky
    # fallaría. np.linalg.det es costoso para n grande, pero sirve
    # como chequeo rápido para este ejercicio.
    if np.linalg.det(ATA) <= 1e-15:
        x_cholesky = np.ones(n - 2) * np.nan   # marcamos como "no calculable"
    else:
        # mi_cholesky devuelve L (triangular inferior) tal que A^T A = L L^T
        L = mi_cholesky(ATA)

        # Resolvemos en dos pasos triangulares:
        #   Paso a) L y = A^T b      (triangular inferior)
        #   Paso b) L^T x = y        (triangular superior)
        # Acá se usa np.linalg.solve porque ya sabemos que L y L^T son
        # triangulares, así que aunque no sea el método más barato,
        # da el resultado correcto (si querés más eficiencia, se puede
        # cambiar por tus funciones sol_trinffil / sol_trsupfil de la Guía 1).
        x_cholesky = np.linalg.solve(L.T, np.linalg.solve(L, ATb))

    # =========================================================
    # 4. MÉTODO 2: TU DESCOMPOSICIÓN QR (sol_cuadmin del Ejercicio 7)
    # =========================================================
    # mi_sol_qr ya hace todo internamente: calcula QR (con permutación
    # de columnas, según implementaste en el ejercicio 4) y resuelve
    # el sistema triangular resultante. Por eso acá solo la llamamos
    # pasando A y b directamente.
    x_qr = mi_sol_qr(A, b)

    # =========================================================
    # 5. COMPARACIÓN DE AMBAS SOLUCIONES
    # =========================================================
    # Ambos métodos resuelven el mismo problema, pero difieren en
    # precisión numérica: las ecuaciones normales elevan el número
    # de condición al cuadrado (cond(A^T A) = cond(A)^2), por lo que
    # amplifican mucho más el error de redondeo que el método QR,
    # sobre todo cuando n crece.
    diferencia = np.linalg.norm(x_cholesky - x_qr)

    return x_cholesky, x_qr, diferencia


# =========================================================
# EJECUCIÓN PARA n = 100 y n = 1000
# =========================================================
if __name__ == "__main__":
    # OJO: acá faltaba la lista de valores de n. Se corrigió a [100, 1000].
    for n in [100, 1000]:
        x_chol, x_qr, diff = ejercicio_9(n)
        print(f"--- Resultados para n = {n} ---")
        print(f"Norma de la diferencia ||x_chol - x_qr||: {diff:.2e}")
        print(f"Primeros 3 elementos de x (QR): {x_qr[:3]}")
        print(f"Últimos 3 elementos de x (QR): {x_qr[-3:]}\n")