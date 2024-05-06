import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

def taylor_polynomial(func, x0, degree, x_values):
    # Definir el símbolo y la función
    x = sp.Symbol('x')
    f = func(x)
    
    # Calcular las derivadas en x0
    derivatives = [f]
    for _ in range(1, degree + 1):
        derivatives.append(derivatives[-1].diff(x))
    
    # Evaluar las derivadas en x0
    derivatives_values = [d.subs(x, x0) for d in derivatives]
    
    # Calcular el polinomio de Taylor
    taylor_poly = sum([(derivatives_values[i] / math.factorial(i)) * (x - x0)**i for i in range(degree + 1)])
    
    # Calcular el error absoluto
    error = f.subs(x, x0) - taylor_poly.subs(x, x0)
    
    # Evaluar el polinomio de Taylor en los valores de x dados
    taylor_values = [taylor_poly.subs(x, val) for val in x_values]
    
    return taylor_values, error

# Ejemplo de uso
x_values = np.linspace(-2, 2, 100)
func = sp.sin  # Función a aproximar (en este caso, sin(x))
x0 = 0  # Punto alrededor del cual se calcula el polinomio de Taylor
degree = 3  # Grado del polinomio de Taylor

taylor_values, error = taylor_polynomial(func, x0, degree, x_values)

# Graficar la función original y el polinomio de Taylor
plt.plot(x_values, np.sin(x_values), label='Función Original')
plt.plot(x_values, taylor_values, label=f'Polinomio de Taylor (grado {degree})')
plt.scatter([x0], [np.sin(x0)], color='red', label=f'Punto de aproximación (x={x0})')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aproximación de Taylor')
plt.grid(True)
plt.show()

print(f'Error absoluto en x={x0}: {abs(error)}')
