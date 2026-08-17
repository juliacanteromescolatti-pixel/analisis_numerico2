import numpy as np
m = 9
n = 6
A = np.random.rand(m,n)
print("Matriz A(9*6) ")
print(A)

# INCISO A
print()
print("INCISO A")
print()
#
  # i
"""Los rangos van de cero a (m-1) y de cero a (n-1)"""
print("I")
print()
i_col = A[n]
i_filas = A[m-1,]
print("El bloque columnas es: ")
print(i_col)
print()
print("El bloque filas es: ")
print(i_filas)
  # ii
print()
print("II")
a_bloq = A[0:m-2,n-1]
# Fijo los elemtos de la columna y elijo los elementos de las filas para esas columnas
b_bloq = A[m-1,0:n-2]
#x_bloq = A[filas, columnas]
print()
print("El bloque A es: ")
print(a_bloq)
print()
print("El bloque B es: ")
print(b_bloq)
print()
  #iii
print("III")
print()
c_bloq = A[n-2:m-2,m-n]
print("El bloque C es: ")
print(c_bloq)
print()
print()
  #iv
print("VI")
print()
d_bloq = A[n:m,2:n]
print("El bloque D es: ")
print(d_bloq)
print()