from agents.unicycle_agents import *
from agent_plotting import *

bea = UnicycleCommander(init_vel=-0.4, init_heading=0., init_heading_dot=0)
bea.create_follower([0., 0.])
bea.followers[0].set_state(vel=.001, heading=np.pi/4)

dt = .01
tf = 15
n = int(tf / dt)

for frame in range(n):
    cv_plot_unicycle_commander_and_followers(bea, dt)
    bea.forward_march(dt)
