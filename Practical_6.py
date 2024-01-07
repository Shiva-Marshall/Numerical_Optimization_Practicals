import numpy as np
import scipy.optimize as opt

def f(x):
  return x**2 - 2*x + 1

# Define the constraint function
def g(x):
  return x - 1

# Solve the constraint optimization problem
x_opt = opt.minimize(f, np.array([1]), constraints={'type': 'ineq', 'fun': g})

# Print the optimal solution
print('The optimal solution is x = {}'.format(x_opt.x))
