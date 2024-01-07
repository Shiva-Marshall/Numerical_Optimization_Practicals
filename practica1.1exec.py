from scipy.optimize import line_search
import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective(x):
    return (-5.0 + x) ** 2.0

# Gradient for the objective function
def gradient(x):
    return 2.0 * (-5.0 + x)

# Define the starting point
point = np.array([-5.0])

# Define the direction to move
direction = np.array([100.0])

# Print the initial conditions
print('start=%.1f, direction=%.1f' % (point[0], direction[0]))

# Perform the line search
result = line_search(objective, gradient, point, direction)

# Extract the alpha (step size)
alpha = result[0]

# Check if the line search was successful
if result[0] is not None:
    print('Alpha: %f' % alpha)
    print('Function evaluations: %d' % result[1])

    # Calculate the point after the line search
    end = point + alpha * direction

    # Evaluate objective function minima
    print('f(end) = f(%.3f) = %.3f' % (end[0], objective(end[0])))

    # Define range
    r_min, r_max = -10.0, 20.0

    # Prepare inputs
    inputs = np.arange(r_min, r_max, 0.1)

    # Compute targets
    targets = [objective(x) for x in inputs]

    # Plot inputs vs objective
    plt.plot(inputs, targets, '--', label='objective')

    # Plot start and end of the search
    plt.plot([point[0]], [objective(point[0])], 's', color='g')
    plt.plot([end[0]], [objective(end[0])], 's', color='r')
    plt.legend()
    plt.show()
else:
    print("Line search did not converge.")
