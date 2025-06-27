import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from agents.point_agents import *
from plotting.agent_plotting import *

# Create the commander and followers
al = PointCommander()
al.create_follower([1, 1.5, 0])
al.create_follower([.8, -1.2, 0])
al.create_follower([-1.3, -.3, 0])
n_agents = 1 + len(al.followers)

dt = .01
tf = 15
n = int(tf / dt)

# Create matrix for the sake of plotting range circles
range_matrix = np.zeros((n_agents, n_agents))
range_matrix[0:1, :] += 1

# Initialize estimates
estimates = np.array([[]])

# Iterate over time
for frame in range(n):

    # Plot and move commander
    cv_plot_commander_and_followers(al, dt, range_rings_indices=range_matrix, estimate=estimates, particles=al.particles)
    al.meander_lead(dt)

    # Estimate the location of the commander
    if frame % 1 == 0:

        estimates = []

        # Estimate commander location
        x_est, y_est = al.localize_particle_filter(al.followers, dt, num_particles=10)
        estimates.append([x_est, y_est])

        estimates = np.array(estimates)
