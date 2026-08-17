# ACT 7
import numpy as np

A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
B = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])

# INCISO A
print("INCISO A")
# Creamos la matriz identidad y la matriz nula para el inciso A
I3 = np.eye(3)
C3 = np.zeros((3,3))
# definimos c

C = np.block([[A, -I3, C3], [-I3, B, -I3], [C3, -I3, A]])

print("La matriz C es: ")
print(C)
print()

# INCISO B
print("INCISO B")
print()

I2 = np.eye(2)
M1 = np.block([[A, -I2], [-I2, B]])
M3 = np.block([[B, -I2], [-I2, A]])

#def fun(A, B):
#X = algo

# INCISO C
print("INCISO C")
print()