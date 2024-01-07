import numpy as np
import matplotlib.pyplot as plt

# Objective function coefficients
c = np.array([3, 2])

# Coefficients matrix for constraints
A = np.array([[1, 2], [2, 1]])

# Right-hand side of constraints
b = np.array([6, 8])

# Solve the linear system to find corner points
corner_points = np.linalg.solve(A, b)

# Define the feasible region
x = np.linspace(0, 5, 100)

# Constraints
eq1 = (6 - A[0, 0] * x) / A[0, 1]
eq2 = (8 - A[1, 0] * x) / A[1, 1]

# Plotting
plt.plot(x, eq1, label=r'$x_1 + 2x_2 \leq 6$')
plt.plot(x, eq2, label=r'$2x_1 + x_2 \leq 8$')

# Highlight corner points
plt.scatter(corner_points[0], corner_points[1], color='red', label='Corner Points')

plt.xlim((0, 5))
plt.ylim((0, 5))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.fill_between(x, np.minimum(eq1, eq2), where=(x >= 0) & (x <= 5), color='gray', alpha=0.3, label='Feasible Region')
plt.legend()
plt.show()
