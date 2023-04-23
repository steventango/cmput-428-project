import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, minimize
from scipy.optimize import NonlinearConstraint as NC

# standard circreg

x = np.linspace(-1, 1, 11)
y = np.sqrt([0, 0.55, 0.8, 0.9, 0.95, 0.95, 1, 0.9, 0.8, 0.7, 0.1])

xc = np.mean(x)
yc = np.mean(y)

def d(c):
  # optimize sum of squares of distances to mean circle
  r = np.sqrt((x-c[0])**2+(y-c[1])**2)
  return r - np.mean(r)

def f(c):
  return np.sum(d(c)**2)

def g(c):
  # constraint
  return np.mean(np.sqrt((x-c[0])**2+(y-c[1])**2)) - np.sqrt((x[-1]-c[0])**2+(y[-1]-c[1])**2)

c = minimize(f, [xc, yc], method="trust-constr", constraints=NC(g, 0, 0)).x
r = np.mean(np.sqrt((x-c[0])**2+(y-c[1])**2))
circ = plt.Circle(c, r, color='r', fill=False)
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.add_patch(circ)
plt.show()
