from point_agents import *
from agent_plotting import *

al = PointCommander()
al.generate_circle_of_followers(10)

dt = .01
tf = 15
n = int(tf / dt)

for frame in range(n):
    cv_plot_commander_and_followers(al, dt)

    if frame * dt < 1:
        # Meander all agents
        al.meander(dt, max=.05)
        for agent in al.followers:
            agent.meander(dt, max=.02)
    else:
        # Meander only the lead agent
        al.meander_lead(dt)
