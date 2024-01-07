import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return -10*np.cos(np.pi*x - 2.2) + (x + 1.5)*x

# Plot the graph of the function
x = np.linspace(-2, 3, 100)
y = f(x)
plt.plot(x, y)

# Identify the local maxima and minima
local_maxima = []
local_minima = []
for i in range(1, len(x) - 1):
  if y[i] > y[i - 1] and y[i] > y[i + 1]:
    local_maxima.append(x[i])
  elif y[i] < y[i - 1] and y[i] < y[i + 1]:
    local_minima.append(x[i])

# Compare the values of the function at the local maxima and minima
global_maximum = max(local_maxima)
global_minimum = min(local_minima)

# Plot the global maximum and minimum
plt.plot(global_maximum, f(global_maximum), 'o', markersize=10, color='red')
plt.plot(global_minimum, f(global_minimum), 'o', markersize=10, color='blue')

# Set labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Global Optimal Solution of f(x) = -10Cos(πx - 2.2) + (x + 1.5)x')

# Show the plot
plt.show()
