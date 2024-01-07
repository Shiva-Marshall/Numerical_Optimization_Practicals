import numpy as np

def f(x):
  return 100 * (x[0]**2 - x[1]**2)**2 + (1 - x[0])**2

def grad_f(x):
  return np.array([
    400 * x[0] * (x[0]**2 - x[1]**2) - 2 * (1 - x[0]),
    -400 * x[1] * (x[0]**2 - x[1]**2)
  ])

def hessian_f(x):
  return np.array([
    [400 * (3 * x[0]**2 - x[1]**2) + 2, -800 * x[0] * x[1]],
    [-800 * x[0] * x[1], 400 * (x[0]**2 - 3 * x[1]**2)]
  ])

# Example usage:

x = np.array([0.5, 0.5])

gradient = grad_f(x)
hessian = hessian_f(x)

print("Gradient:", gradient)
print("Hessian:", hessian)
