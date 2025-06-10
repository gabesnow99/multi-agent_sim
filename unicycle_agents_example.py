from agents.unicycle_agents import *
from agent_plotting import *

bea = UnicycleCommander(init_vel=0.001, init_heading=np.pi/4., init_heading_dot=0.)
bea.generate_circle_of_followers(8, r=0.)

for f in bea.followers:
    f.set_state(vel=.01, pos=[1, -1])

dt = .01
tf = 25
n = int(tf / dt)

for frame in range(n):
    cv_plot_unicycle_commander_and_followers(bea, dt)
    # bea.meander_lead(dt)
    bea.forward_march(dt)
