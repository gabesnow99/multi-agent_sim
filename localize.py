'''
THE PURPOSE OF THIS FILE WILL BE TO USE RANGE ONLY LOCALIZATION TO CREATE A MAP
OF THE COMMANDER AGENT AND ITS FOLLOWERS AND ESTIMATE THE LOCATION OF THE COMMANDER.
'''

import numpy as np

from agents.point_agents import *
from plotting.agent_plotting import *

from probability_visualization.probability_representations import Donut, MultiDonut

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
range_matrix[0, :] += 1

# Initialize estimates
estimates = np.array([[]])

# Iterate over time
for frame in range(n):

    # Plot and move commander
    cv_plot_commander_and_followers(al, dt, range_rings_indices=range_matrix, estimate=estimates)
    al.meander_lead(dt)

    # Estimate the location of the commander
    if frame % 2 == 0:
        donuts = []
        for follower in al.followers:
            donut = Donut(follower.range_to_agent(al), .05, follower.pos[0], follower.pos[1], num_points=10000)
            donuts.append(donut)
        md = MultiDonut(donuts, .02, .02)
        x_est, y_est, z_est = md.get_max_loc()
        estimates = np.array([[x_est, y_est]])
