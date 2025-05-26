from agents.unicycle_agents import *
from agent_plotting import *

bea = UnicycleAgent()

dt = .01
tf = 15
n = int(tf / dt)

for frame in range(n):
    cv_plot_unicycle_agent(bea, dt)
    bea.meander(dt)