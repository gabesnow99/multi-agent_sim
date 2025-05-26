import numpy as np
import random

# Super class of unicycle model agents
class UnicycleAgent:

    '''
    self.pos (1x2 Array):       position (m) as [x, y]
    self.vel (Float):           velocity (m/s), in the direction of self.heading
    self.heading (Float):       direction (rad), between [-pi, pi]
    self.heading_dot (Float):   change in heading with respect to time (rad/sec)
    '''

    def __init__(self, init_pos=[0., 0.], init_vel=0., init_heading=0., init_heading_dot=0., mass=0.):
        self.pos = np.array(init_pos).flatten()
        self.vel = init_vel
        self.heading = self.wrap(init_heading)
        self.heading_dot = init_heading_dot
        self.mass = mass

    # Manually set the state
    def set_state(self, pos=None, vel=None, heading=None, heading_dot=None):
        if pos:
            self.pos = pos
        if vel:
            self.vel = vel
        if heading:
            self.heading = self.wrap(heading)
        if heading_dot:
            self.heading_dot = heading_dot

    # Move forward a timestep
    def propagate(self, dt):
        self.pos += dt * self.vel * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.heading = self.wrap(self.heading + dt * self.heading_dot)

    # Add a random amount to the velocities
    def meander(self, dt, v_max=.1, w_max=.03):
        self.vel += random.uniform(-v_max, v_max)
        self.heading_dot += random.uniform(-w_max, w_max)
        self.propagate(dt)

    # Wraps angles between -pi and pi
    def wrap(self, theta):
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < np.pi:
            theta += 2 * np.pi
        return theta
