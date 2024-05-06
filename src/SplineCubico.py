import numpy as np
import matplotlib.pyplot as plt

def tridiag_solver(A, b):
    n = len(b)
    x = np.zeros(n)

    # Forward elimination
    for i in range(1, n):
        m = A[i][i-1] / A[i-1][i-1]
        A[i][i] -= m * A[i-1][i]
        b[i] -= m * b[i-1]

    # Back substitution
    x[-1] = b[-1] / A[-1][-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - A[i][i+1] * x[i+1]) / A[i][i]

    return x

def spline_cubico_natural(x, y, x_new):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h

    # Construir la matriz tridiagonal
    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0][0] = 1
    A[-1][-1] = 1
    for i in range(1, n-1):
        A[i][i-1] = h[i-1]
        A[i][i] = 2 * (h[i-1] + h[i])
        A[i][i+1] = h[i]
        b[i] = 3 * (delta[i] - delta[i-1])

    # Resolver el sistema de ecuaciones lineales
    c = tridiag_solver(A, b)

    # Calcular los coeficientes restantes
    d = np.diff(c) / (3 * h)
    b = delta - h * (2 * c[:-1] + c[1:]) / 3
    a = y[:-1]

    # Evaluar el spline en los nuevos puntos
    y_new = np.zeros_like(x_new)
    for i, xi in enumerate(x_new):
        idx = np.searchsorted(x, xi)
        if idx == 0:
            idx += 1
        elif idx == n:
            idx -= 1
        t = xi - x[idx-1]
        y_new[i] = a[idx-1] + b[idx-1] * t + c[idx-1] * t**2 + d[idx-1] * t**3

    return y_new

def spline_cubico_sujeta(x, y, x_new, derivada_primera_inicio, derivada_primera_final):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h

    # Construir la matriz tridiagonal
    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0][0] = 2 * h[0]
    A[0][1] = h[0]
    b[0] = 3 * (delta[0] - derivada_primera_inicio)

    A[-1][-2] = h[-1]
    A[-1][-1] = 2 * h[-1]
    b[-1] = 3 * (derivada_primera_final - delta[-1])

    for i in range(1, n-1):
        A[i][i-1] = h[i-1]
        A[i][i] = 2 * (h[i-1] + h[i])
        A[i][i+1] = h[i]
        b[i] = 3 * (delta[i] - delta[i-1])

    # Resolver el sistema de ecuaciones lineales
    c = tridiag_solver(A, b)

    # Calcular los coeficientes restantes
    d = np.diff(c) / (3 * h)
    b = delta - h * (2 * c[:-1] + c[1:]) / 3
    a = y[:-1]

    # Evaluar el spline en los nuevos puntos
    y_new = np.zeros_like(x_new)
    for i, xi in enumerate(x_new):
        idx = np.searchsorted(x, xi)
        if idx == 0:
            idx += 1
        elif idx == n:
            idx -= 1
        t = xi - x[idx-1]
        y_new[i] = a[idx-1] + b[idx-1] * t + c[idx-1] * t**2 + d[idx-1] * t**3

    return y_new


# Ejemplo de uso y graficado
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 0, 1, 0, 1])  # Valores de ejemplo
x_new = np.linspace(0, 5, 100)

# Splines cúbicos con diferentes condiciones de frontera
y_spline_natural = spline_cubico_natural(x, y, x_new)
y_spline_sujeta = spline_cubico_sujeta(x, y, x_new, derivada_primera_inicio=0, derivada_primera_final=0.5)

# Graficar los puntos originales y los splines cúbicos
plt.plot(x_new, y_spline_natural, label='Spline Cúbico Natural')
plt.plot(x_new, y_spline_sujeta, label='Spline Cúbico Sujeta')
plt.scatter(x, y, c='red', label='Puntos Originales')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Splines Cúbicos')
plt.grid(True)
plt.show()
