import numpy as np 

def egaussp(A, b):
    m, n = A.shape
    U = A.copy()
    y = b.copy()

#Como no se si m es mayor o menor que n, defino el for asi para cubrir todoslos casos.
    for k in range(min(m - 1, n)):
        # si la columna son ceros no hace aplicar gauss
        if np.argmax(np.abs(U[k:, k])) != 0:
            # Elegimos el pivot
            l = k + np.argmax(np.abs(U[k:, k]))
            # Pivoteamos (sin multiplicar por matriz elemental)
            U[[k, l], :] = U[[l, k], :] #hacemos el cambio de filas
            y[[k, l]] = y[[l, k]] #cambio tambien los vectores
            # Aplicamos las tranformaciones de gauss
            v = U[k + 1:, k] / U[k, k] #defino a v en ssel indice I
            U[k + 1:, k] = 0
            U[k + 1:, k + 1:] = U[k + 1:, k + 1:] - np.outer(v, U[k, k + 1:]) #Es la misma matriz menos el producto exterior para obtener una matriz
            y[k + 1:] = y[k + 1:] - v * y[k] #uso este producto pues uno es un vector y otro es un nro

    return U, y

#TEST 
A= np.random.rand(5,5)
b = np.random.rand(5)
print("La solucion con egaussp es:")
U, y = egaussp(A,b)

print(f'U= {U}')
print(f'y= {y}')

import numpy as np 

def dlup(A):
    n = A.shape[0]
    U = A.copy()
    P = np.eye(n)
    for k in range(n): 
        pivot = k + np.argmax(np.abs(U[k:, k])) #np.armax no solo me die el max en valor abs sino tambien donde esta
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
        
        U[k+1: , k] = U[k+1:, k]/U[k, k] 
        U[k+1: , k +1:] = U[k+1: , k+1:]-np.outer(U[k+1: , k], U[k, k+1:])
        
    L = np.tril(U,-1)+np.eye(n)
    U = np.triu(U)

    return U, L, P 

# #TEST 
A= np.random.rand(5,5)
print("La solucion con dlup es:")
U, L, P = dlup(A)
print(f'U={U}')
print(f'A={L}')
print(f'P={P}')
print(P.T@L@U-A)

            