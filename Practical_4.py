import numpy as np

def f(x):
    return -10*np.cos(np.pi*x - 2.2) + (x + 1.5)*x

# Identify the local maxima and minima
def find_optimal_solution():
    x_values = np.linspace(-2, 3, 1000)
    y_values = f(x_values)

    local_maxima = []
    local_minima = []

    for i in range(1, len(x_values) - 1):
        if y_values[i] > y_values[i - 1] and y_values[i] > y_values[i + 1]:
            local_maxima.append(x_values[i])
        elif y_values[i] < y_values[i - 1] and y_values[i] < y_values[i + 1]:
            local_minima.append(x_values[i])

    # Compare the values of the function at the local maxima and minima
    global_maximum = max(local_maxima)
    global_minimum = min(local_minima)

    return global_maximum, f(global_maximum), global_minimum, f(global_minimum)

# Find and print the global optimal solution
global_maximum_x, global_maximum_y, global_minimum_x, global_minimum_y = find_optimal_solution()
print(f'Global Maximum: x = {global_maximum_x}, f(x) = {global_maximum_y}')
print(f'Global Minimum: x = {global_minimum_x}, f(x) = {global_minimum_y}')
