import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify

def lagrange_interpolation(points, x):
    n = len(points)
    polynomial = 0

    for i in range(n):
        numerator = 1
        denominator = 1
        xi, yi = points[i]

        for j in range(n):
            if i != j:
                xj, _ = points[j]
                numerator *= (x - xj)
                denominator *= (xi - xj)

        polynomial += yi * (numerator / denominator)

    return polynomial

# Puntos a interpolar
points = [(-1, np.sin(np.pi * -1)),
          (-1/2, np.sin(np.pi * (-1/2))),
          (0, np.sin(np.pi * 0)),
          (1/2, np.sin(np.pi * (1/2))),
          (1, np.sin(np.pi * 1))]

# Función a comparar
expression = "sin(pi*x)"
x = symbols('x')
function = lambdify(x, expression)

# Generación de puntos para graficar la función de interpolación
x_interp = np.linspace(-1, 1, 100)
y_interp = np.array([lagrange_interpolation(points, xi) for xi in x_interp])

# Generación de puntos para graficar la función ingresada
x_func = np.linspace(-1, 1, 100)
y_func = function(x_func)

# Graficación de la función de interpolación y la función ingresada
plt.plot(x_interp, y_interp, label="Interpolación de Lagrange")
plt.plot(x_func, y_func, label="Función ingresada")
plt.scatter([x for x, _ in points], [y for _, y in points], c="red", label="Puntos conocidos")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolación de Lagrange vs Función ingresada")
plt.grid(True)
plt.show()

