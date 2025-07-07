import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from agents.point_agents import *
from plotting.agent_plotting import *

# Create the commander and followers
al = PointCommander(init_vel=[.5, .8, 0.])
al.create_follower([1, 1.5, 0])
al.create_follower([.8, -1.2, 0])
al.create_follower([-1.3, -.3, 0])
n_agents = 1 + len(al.followers)

dt = .01
tf = 1
n = int(tf / dt)

# Create matrix for the sake of plotting range circles
range_matrix = np.zeros((n_agents, n_agents))
range_matrix[0:1, :] += 1

# Initialize estimates
estimates = np.array([[]])

# Iterate over time
for frame in range(n):

    # Plot and move commander
    cv_plot_commander_and_followers(al, dt, range_rings_indices=range_matrix, estimate=estimates)
    # al.meander_lead(dt)
    # al.forward_march(dt)

    # Estimate the location of the commander
    if frame % 1 == 0:

        estimates = []

        # Estimate commander location
        # al.optimized_localization(al.followers, dt)
        estimates.append(al.estimated_pos[:2].tolist())

        # Estimate follower locations
        for ii, agent_to_estimate in enumerate(al.followers):
            to_range = []
            to_range.append(al)
            for jj, follower in enumerate(al.followers):
                if jj != ii:
                    to_range.append(follower)
            if ii in [0, 1, 2]:
                agent_to_estimate.optimized_localization(to_range, dt)
            estimates.append(agent_to_estimate.estimated_pos[:2].tolist())

        estimates = np.array(estimates)
