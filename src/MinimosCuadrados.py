import numpy as np
import matplotlib.pyplot as plt

def minimos_cuadrados(x, y):
    if len(x) != len(y):
        print('[ERROR]: X e Y tienen que tener la misma cantidad de puntos')
        return
    
    n = len(x)

    # Calculamos a y b  
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x ** 2)
    sum_xy = np.sum(x * y) 

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    print(f'[CONSOLE]: El valor de m es: {m}.\n[CONSOLE]: El valor de b es: {b}')

    # Graficar los puntos y la línea de ajuste
    plt.scatter(x, y, label='Puntos')
    plt.plot(x, m*x + b, color='red', label='Ajuste de mínimos cuadrados')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ajuste de mínimos cuadrados')
    plt.legend()
    plt.grid(True)
    plt.show()

    y_predictor = m * x + b

    r_cuadrado = np.sum((y_predictor - np.mean(y)) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f'[CONSOLE]: Coeficiente de determinación (R^2): {r_cuadrado}')
               
    return m, b

# Ejemplo de uso
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3.5, 5, 4.5, 6])

minimos_cuadrados(x, y)
