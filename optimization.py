import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from plotting.agent_plotting import range_to_pixels, pos_to_pixels, red, green, blue

c1 = [2, 2]
r1 = 3
c2 = [-1, 2]
r2 = 2
c3 = [-1.2, -1.5]
r3 = 2.3

dx, dy = .1, .1
x_min, x_max, y_min, y_max = -5., 5., -5., 5.

ranges = np.array([r1, r2, r3])
circes = np.array([c1, c2, c3])
def calc_cost(x_, y_):
    dists = np.array([np.linalg.norm(np.array(circ) - np.array([x_, y_])) for circ in circes])
    errs = np.abs(ranges - dists)
    cost = 0.
    # cost += np.sum(errs)
    # cost += np.prod(errs)
    cost += np.dot(errs, errs)
    return cost

d = {}
for ii in np.linspace(x_min, x_max, int((x_max - x_min) / dx) + 1):
    for jj in np.linspace(y_min, y_max, int((y_max - y_min) / dy) + 1):
        cost = calc_cost(ii, jj)
        weight = 1. / cost
        d[(ii, jj)] = weight

def sum_squared(xy):
    dists = np.array([np.linalg.norm(np.array(circ) - np.array(xy)) for circ in circes])
    errs = np.abs(ranges - dists)
    cost = 0.
    cost += np.dot(errs, errs)
    return cost

def sum_product(xy):
    dists = np.array([np.linalg.norm(np.array(circ) - np.array(xy)) for circ in circes])
    errs = np.abs(ranges - dists)
    cost = 0.
    cost += np.sum(errs)
    cost += np.prod(errs)
    return cost

bounds = [(x_min, x_max), (y_min, y_max)]
result_sum_squared = differential_evolution(sum_squared, bounds)
result_sum_product = differential_evolution(sum_product, bounds)
print("Best parameters sum squared:", result_sum_squared.x)
print("Best parameters sum product:", result_sum_product.x)

# Points for plotting
x_vals = [point[0] for point in d.keys()]
y_vals = [point[1] for point in d.keys()]
z_vals = np.array(list(d.values()))

# 3D bar plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x_vals, y_vals, [0]*len(z_vals), dx=dx, dy=dy, dz=z_vals)
# plt.show()

# Circles
pixels_per_meter = 100
width, height = int(pixels_per_meter * (x_max - x_min)), int(pixels_per_meter * (y_max - y_min))
img = np.full((height, width, 3), 255, dtype=np.uint8)
cv2.circle(img, pos_to_pixels(c1[0], c1[1]), range_to_pixels(r1), red, 2)
cv2.circle(img, pos_to_pixels(c2[0], c2[1]), range_to_pixels(r2), red, 2)
cv2.circle(img, pos_to_pixels(c3[0], c3[1]), range_to_pixels(r3), red, 2)

x_ss, y_ss = result_sum_squared.x
x_sp, y_sp = result_sum_product.x
cv2.circle(img, pos_to_pixels(x_ss, y_ss), 6, green, -1)
cv2.circle(img, pos_to_pixels(x_sp, y_sp), 6, blue, -1)

cv2.imshow('Agents', img)
if cv2.waitKey(int(30000)) & 0xFF == ord('q'):
    quit()
