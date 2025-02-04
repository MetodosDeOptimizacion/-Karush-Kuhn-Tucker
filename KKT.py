import numpy as np
from scipy.optimize import minimize

# -------------------------------------------------------------
# Ejercicio 1: Minimización con una restricción de desigualdad
# -------------------------------------------------------------

def f1(x):
    """
    Función objetivo para el Ejercicio 1: f(x1, x2) = x1^2 + 2*x2^2
    """
    return x[0]**2 + 2*x[1]**2

def constraint1(x):
    """
    Restricción para el Ejercicio 1: x1 + 2x2 - 3 <= 0
    """
    return 3 - (x[0] + 2*x[1])

def solve_exercise_1():
    """
    Resolver el Ejercicio 1 utilizando el método SLSQP de scipy.optimize.
    """
    # Estimación inicial para las variables x1 y x2
    x0 = np.array([0.5, 0.5])
    
    # Definir las restricciones como diccionarios
    cons = ({'type': 'ineq', 'fun': constraint1})
    
    # Ejecutar la optimización
    res = minimize(f1, x0, constraints=cons, method='SLSQP')
    
    # Retornar los resultados de la optimización
    return res.x, res.fun, res.success

# -------------------------------------------------------------
# Ejercicio 2: Minimización con restricciones múltiples
# -------------------------------------------------------------

def f2(x):
    """
    Función objetivo para el Ejercicio 2: f(x1, x2) = x1^2 + x2^2
    """
    return x[0]**2 + x[1]**2

def constraint2(x):
    """
    Restricción 1 para el Ejercicio 2: x1 + x2 - 2 <= 0
    """
    return 2 - (x[0] + x[1])

def constraint3(x):
    """
    Restricción 2 para el Ejercicio 2: x1 >= 0
    """
    return x[0]

def solve_exercise_2():
    """
    Resolver el Ejercicio 2 utilizando el método SLSQP de scipy.optimize.
    """
    # Estimación inicial para las variables x1 y x2
    x0 = np.array([1.0, 1.0])
    
    # Definir las restricciones como diccionarios
    cons = (
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3}
    )
    
    # Ejecutar la optimización
    res = minimize(f2, x0, constraints=cons, method='SLSQP')
    
    # Retornar los resultados de la optimización
    return res.x, res.fun, res.success

# -------------------------------------------------------------
# Ejercicio 3: Maximización con restricciones
# -------------------------------------------------------------

def f3(x):
    """
    Función objetivo para el Ejercicio 3 (Maximización): f(x1, x2) = 3x1 + 4x2
    Se minimiza el negativo de la función para obtener la maximización.
    """
    return -(3*x[0] + 4*x[1])

def constraint4(x):
    """
    Restricción para el Ejercicio 3: x1^2 + x2^2 <= 9
    """
    return 9 - (x[0]**2 + x[1]**2)

def constraint5(x):
    """
    Restricción para el Ejercicio 3: x1 >= 0
    """
    return x[0]

def constraint6(x):
    """
    Restricción para el Ejercicio 3: x2 >= 0
    """
    return x[1]

def solve_exercise_3():
    """
    Resolver el Ejercicio 3 utilizando el método SLSQP de scipy.optimize.
    """
    # Estimación inicial para las variables x1 y x2
    x0 = np.array([2.0, 2.0])
    
    # Definir las restricciones como diccionarios
    cons = (
        {'type': 'ineq', 'fun': constraint4},
        {'type': 'ineq', 'fun': constraint5},
        {'type': 'ineq', 'fun': constraint6}
    )
    
    # Ejecutar la optimización
    res = minimize(f3, x0, constraints=cons, method='SLSQP')
    
    # Retornar los resultados de la optimización (valor negativo revertido para maximización)
    return res.x, -res.fun, res.success

# -------------------------------------------------------------
# Función Principal: Ejecutar todos los ejercicios y mostrar los resultados
# -------------------------------------------------------------

def main():
    """
    Ejecuta los tres ejercicios de optimización y muestra los resultados obtenidos.
    """
    print("Resolviendo Ejercicios de Optimización usando las Condiciones KKT\n")
    
    # Ejercicio 1
    print("Ejercicio 1 - Minimización con restricción de desigualdad:")
    x1, f1_value, success1 = solve_exercise_1()
    print(f"Solución: x1 = {x1[0]:.4f}, x2 = {x1[1]:.4f}")
    print(f"Valor de la función objetivo: f(x1, x2) = {f1_value:.4f}")
    print(f"¿Optimización exitosa?: {'Sí' if success1 else 'No'}\n")
    
    # Ejercicio 2
    print("Ejercicio 2 - Minimización con múltiples restricciones:")
    x2, f2_value, success2 = solve_exercise_2()
    print(f"Solución: x1 = {x2[0]:.4f}, x2 = {x2[1]:.4f}")
    print(f"Valor de la función objetivo: f(x1, x2) = {f2_value:.4f}")
    print(f"¿Optimización exitosa?: {'Sí' if success2 else 'No'}\n")
    
    # Ejercicio 3
    print("Ejercicio 3 - Maximización con restricciones:")
    x3, f3_value, success3 = solve_exercise_3()
    print(f"Solución: x1 = {x3[0]:.4f}, x2 = {x3[1]:.4f}")
    print(f"Valor de la función objetivo (maximización): f(x1, x2) = {f3_value:.4f}")
    print(f"¿Optimización exitosa?: {'Sí' if success3 else 'No'}\n")

# Ejecutar el programa
if __name__ == "__main__":
    main()
