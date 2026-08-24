import numpy as np
"""
    Recibe los coeficientes de un polinomio f(x) = c_0 + c_1*x + ... + c_d*x^d
    y una matriz X simétrica semidefinida positiva.
    Devuelve la matriz Y tal que Y_ij = f(X_ij) evaluado elemento a elemento.
"""
def pol_matrix(c, X):
    """
    Versión explícita elemento a elemento.
    Recibe la lista de coeficientes 'c' y la matriz 'X'.
    """
    X = np.array(X, dtype=float)
    filas = X.shape[0]
    columnas = X.shape[1]
#Averigua cuántas filas y cuántas columnas tiene la matriz para saber cuántas veces va a tener que repetir el cálculo.
    
    # Creamos una matriz vacía de ceros del mismo tamaño para guardar las respuestas
    Y = np.zeros((filas, columnas))
    
    # Recorremos cada fila de la matriz
    for i in range(filas):
        # Recorremos cada columna de la matriz
        for j in range(columnas):
            
            # Agarramos el número individual que está en la posición (i, j)
            valor_individual = X[i, j]
#Andá a la matriz X, buscá el número que está justo en el cruce de la fila i y la columna j, 
#sacalo de ahí y guardalo momentáneamente en una variable llamada valor_individual".
#Sirve para convertir una operación de matrices en una operación de números comunes (escalares).

            # Calculamos el polinomio tradicional para ESTE número:
            # f(x) = c_0 + c_1*x + c_2*x^2 + ...
            resultado_polinomio = 0
            for grado in range(len(c)):
                coef = c[grado]
                resultado_polinomio += coef * (valor_individual ** grado)
            
            # Guardamos el resultado final en la posición correspondiente de Y
            Y[i, j] = resultado_polinomio
            
    return Y
"""
EXPLICACION ULTIMO BUCLE:
Una vez que tiene el número aislado (por ejemplo, el 2.0), entra al tercer bucle:
for grado in range(len(c)).Este bucle recorre las posiciones de la lista de 
coeficientes.coef = c[grado]: Va agarrando uno a uno los coeficientes
(c_0, luego c_1, luego c_2).resultado_polinomio += coef * (valor_individual ** grado):
Va acumulando la suma de la estructura matemática:Resultado = c_0*x^0 + c_1*x^1 + c_2*x^2 +... 
Al terminar con esa celda, guarda el valor final en Y[i, j] y pasa a la siguiente combinación de fila y columna.
"""


#EJECUCIÓN DIRECTA
# Definimos una matriz X simétrica semidefinida positiva
X_prueba = np.array([[2.0, 1.0], [1.0, 2.0]])

# Coeficientes del polinomio f(x) = 1 + 2x + 3x^2 -> c = [1, 2, 3]
coeficientes = [1, 2, 3]

# Calculamos Y = f(X) elemento a elemento
Y_resultado = pol_matrix(coeficientes, X_prueba)

print("EJERCICIO 10")
print("Matriz original X:")
print(X_prueba)
print("\nMatriz resultante Y = f(X) elemento a elemento:")
print(Y_resultado)

# Validación numérica rápida de que Y es semidefinida positiva (autovalores >= 0)
autovalores = np.linalg.eigvals(Y_resultado)
print(f"\nAutovalores de Y para verificar consistencia: {autovalores}")