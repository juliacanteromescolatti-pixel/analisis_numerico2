import numpy as np
import matplotlib.pyplot as plt

def graficar_transformacion_esfera(epsilons):
    # Generar la bola unidad B = {x in R^2 : ||x||_2 = 1}
    theta = np.linspace(0, 2 * np.pi, 300)
    # Puntos de la esfera unidad (circunferencia en R^2)
    X = np.array([np.cos(theta), np.sin(theta)]) 

    plt.figure(figsize=(9, 7))
    
    # Graficar la esfera unidad original
    plt.plot(X[0, :], X[1, :], 'k--', label='Esfera Unidad Orig. (Norma 2)', linewidth=2)

    # Graficar la transformación A(eps) * X para cada epsilon
    for eps in epsilons:
        A_eps = np.array([[1.0, 1.0 - eps],
                          [0.0, 1.0]])
        
        # Transformación de los puntos
        Y = A_eps @ X  
        plt.plot(Y[0, :], Y[1, :], label=f'Transformación $\epsilon = {eps}$')

    plt.axhline(0, color='gray', linestyle=':', alpha=0.6)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.6)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.title(r'Transformación de la esfera unidad por $A(\epsilon)$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.show()

# Ejecución para los valores de epsilon indicados en la consigna
epsilons = [0.25, 0.125, 0.0625, 1e-5]
graficar_transformacion_esfera(epsilons)