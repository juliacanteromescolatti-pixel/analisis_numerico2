import numpy as np
import time
import matplotlib.pyplot as plt
from P2_ej11 import sol_egauss  # Importamos tu función de eliminación gaussiana ya hecha

def ajustar_esfera(puntos):
    """
    APARTADOS A y B: Construcción y resolución del sistema lineal.
    La ecuación de la esfera es: x² + y² + z² + Dx + Ey + Fz + G = 0
    Al reemplazar cada punto (x, y, z) se obtiene un sistema lineal M * p = b,
    donde las incógnitas p son [D, E, F, G].
    """
    # Creamos una matriz vacía de 4x4 y un vector de resultados de 4 elementos
    M = np.zeros((4, 4))
    b = np.zeros(4)
    
    # Llenamos el sistema lineal reemplazando las coordenadas de cada uno de los 4 puntos
    for i in range(4):
        x, y, z = puntos[i]
        M[i] = [x, y, z, 1]              # Fila de coeficientes para D, E, F y G
        b[i] = -(x**2 + y**2 + z**2)     # Término independiente del lado derecho
        
    # Resolvemos el sistema 4x4 usando tu función importada del Ejercicio 11
    parametros = sol_egauss(M, b)
    
    # Desempaquetamos los 4 valores calculados (D, E, F, G)
    D, E, F, G = parametros[0], parametros[1], parametros[2], parametros[3]
    return D, E, F, G

def graficar_esfera_completa(puntos, D, E, F, G):
    """
    APARTADO C: Gráfico en 3D de la esfera, los 4 puntos y sus círculos máximos.
    """
    # --------------------------------------------------------------------------
    # BLOQUE 1: CÁLCULO GEOMÉTRICO DEL CENTRO Y EL RADIO
    # Pasamos de la ecuación general al centro físico (h, k, l) y al radio R.
    # --------------------------------------------------------------------------
    h = -D / 2
    k = -E / 2
    l = -F / 2
    R = np.sqrt(h**2 + k**2 + l**2 - G)
    
    # --------------------------------------------------------------------------
    # BLOQUE 2: CONFIGURACIÓN DEL LIENZO TRIDIMENSIONAL
    # Le indicamos a la librería que cree una ventana con volumen 3D (ejes X, Y, Z).
    # --------------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # --------------------------------------------------------------------------
    # BLOQUE 3: DIBUJAR LA PELOTA DE FONDO (Esfera fantasma)
    # Generamos ángulos de latitud y longitud (u, v) para calcular miles de puntos.
    # Usamos alpha=0.15 (15% de opacidad) para que sea translúcida y nos permita
    # ver los alfileres y aros que graficaremos en su interior.
    # --------------------------------------------------------------------------
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x_esf = h + R * np.outer(np.cos(u), np.sin(v))
    y_esf = k + R * np.outer(np.sin(u), np.sin(v))
    z_esf = l + R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_esf, y_esf, z_esf, color='cyan', alpha=0.15, edgecolor='none')
    
    # --------------------------------------------------------------------------
    # BLOQUE 4: MARCAR LOS 4 PUNTOS DADOS (Alfileres rojos)
    # Dibujamos las coordenadas originales como puntos rojos grandes ('s=50').
    # Sirve para corroborar que la esfera pasa exactamente por ellos.
    # --------------------------------------------------------------------------
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], color='red', s=50, label='Puntos dados')
    
    # --------------------------------------------------------------------------
    # BLOQUE 5: TRAZAR LOS CÍRCULOS MÁXIMOS (Aros de hula-hula rotados)
    # Un círculo máximo pasa obligatoriamente por el centro de la esfera. 
    # Calculamos la inclinación tridimensional (ángulos theta y phi) de cada punto 
    # con respecto al centro y rotamos una circunferencia de radio R hacia él.
    # --------------------------------------------------------------------------
    t = np.linspace(0, 2 * np.pi, 100) # El giro de 360 grados para dibujar el aro
    colores = ['blue', 'orange', 'green', 'purple']
    
    for i in range(4):
        # Calculamos la distancia en cada eje (X, Y, Z) desde el centro al punto 'i'
        dx = puntos[i, 0] - h
        dy = puntos[i, 1] - k
        dz = puntos[i, 2] - l
        dist_horizontal = np.hypot(dx, dy) # Distancia proyectada en el piso (plano XY)
        
        # Obtenemos los ángulos de inclinación usando trigonometría (arco tangente)
        theta = np.arctan2(dy, dx) if dist_horizontal > 1e-9 else 0
        phi = np.arctan2(dist_horizontal, dz) if dist_horizontal > 1e-9 else 0
        
        # Aplicamos las matrices de rotación estándar para inclinar el círculo hacia el punto
        cx = h + R * (np.cos(t) * np.cos(theta) * np.cos(phi) - np.sin(t) * np.sin(theta))
        cy = k + R * (np.cos(t) * np.sin(theta) * np.cos(phi) + np.sin(t) * np.cos(theta))
        cz = l - R * np.cos(t) * np.sin(phi)
        
        # Dibujamos la circunferencia en el espacio como una línea discontinua ('--')
        ax.plot(cx, cy, cz, color=colores[i], linestyle='--', alpha=0.8, label=f'Círculo Máximo P{i+1}')

    # --------------------------------------------------------------------------
    # BLOQUE 6: DETALLES VISUALES Y APERTURA DE LA VENTANA
    # Configuramos los textos de los ejes y mostramos la pantalla interactiva.
    # --------------------------------------------------------------------------
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    ax.set_title('Esfera ajustada y sus Círculos Máximos')
    ax.legend()
    plt.show() # Abre el gráfico dinámico (puedes arrastrarlo con el mouse para girarlo)


# --- SCRIPT DE PRUEBA Y EJECUCIÓN ---
if __name__ == "__main__":
    # Definimos 4 puntos cualesquiera en R³ que no sean coplanares (no alineados en una mesa chata)
    puntos_ejemplo = np.array([
        [6.0, 2.0, 3.0],
        [1.0, 7.0, 3.0],
        [1.0, 2.0, 8.0],
        [1.0, 5.0, 7.0]
    ])
    
    print("=== EJECUCIÓN COMPLETA DEL EJERCICIO 15 ===")
    
    # 1. Llamamos a la función matemática para obtener las incógnitas D, E, F, G
    D, E, F, G = ajustar_esfera(puntos_ejemplo)
    
    # 2. Imprimimos los coeficientes por consola
    print(f"Valores obtenidos:\n D = {D:.4f}\n E = {E:.4f}\n F = {F:.4f}\n G = {G:.4f}")
    print(f"\nEcuación general resultante:")
    print(f"x² + y² + z² + ({D:.2f})x + ({E:.2f})y + ({F:.2f})z + ({G:.2f}) = 0")
    
    # 3. Enviamos los datos procesados a la función de dibujo interactivo
    graficar_esfera_completa(puntos_ejemplo, D, E, F, G)
