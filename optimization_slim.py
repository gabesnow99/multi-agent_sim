import numpy as np
from scipy.optimize import differential_evolution

c1 = [2, 2]
r1 = 3
c2 = [-1, 2]
r2 = 2
c3 = [-1.2, -1.5]
r3 = 2.3
ranges = np.array([r1, r2, r3])
circes = np.array([c1, c2, c3])
bounds = [(-5, 5), (-5, 5)]

def calc_cost(xy):
    dists = np.array([np.linalg.norm(np.array(circ) - np.array(xy)) for circ in circes])
    errs = np.abs(ranges - dists)
    cost = 0.
    cost += np.sum(errs)
    cost += np.prod(errs)
    return cost

result = differential_evolution(calc_cost, bounds)
print("Best parameters:", result.x)
print("Maximum value:", -result.fun)