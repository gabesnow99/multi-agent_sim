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

    def __init__(self, init_pos=[0., 0.], init_vel=0., init_heading=0., init_heading_dot=0., mass=1., moment_inertia=1.):
        self.pos = np.array(init_pos).flatten()
        self.vel = init_vel
        self.heading = self.wrap(init_heading)
        self.heading_dot = init_heading_dot
        self.mass = mass
        self.moment_inertia = moment_inertia

    # Manually set the state
    def set_state(self, pos=None, vel=None, heading=None, heading_dot=None):
        if pos != None:
            self.pos = pos
        if vel != None:
            self.vel = vel
        if heading != None:
            self.heading = self.wrap(heading)
        if heading_dot != None:
            self.heading_dot = heading_dot

    # Move forward a timestep
    def propagate(self, dt, F=0., tau=0.):
        self.vel += dt * F / self.mass
        self.heading_dot += dt * tau / self.moment_inertia
        self.pos += dt * self.vel * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.heading = self.wrap(self.heading + dt * self.heading_dot)

    # Add a random amount to the velocities
    def meander(self, dt, v_max=.1, w_max=.1):
        self.vel += random.uniform(-v_max, v_max)
        self.heading_dot += random.uniform(-w_max, w_max)
        self.propagate(dt)

    # Wraps angles between -pi and pi
    def wrap(self, theta):
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta

# Commander agent
class UnicycleCommander(UnicycleAgent):

    def __init__(self, init_pos=[0., 0.], init_vel=0., init_heading=0., init_heading_dot=0., mass=1., moment_inertia=.1):
        super().__init__(init_pos, init_vel, init_heading, init_heading_dot, mass, moment_inertia)
        self.followers = []

    def create_follower(self, rel_pos):
        self.followers.append(UnicycleFollower(commander=self, rel_pos=rel_pos))

    # Generates n followers cluster in a circle of radius r about the commander
    def generate_circle_of_followers(self, n=4, r=1.):
        n = int(n)
        for ii in range(n):
            th = 2 * np.pi * ii / n
            self.create_follower(r * np.array([np.cos(th), np.sin(th)]))

    # Marches the commander as well as all followers
    def forward_march(self, dt, F=0., tau=0.):
        self.propagate(dt, F, tau)
        for agent in self.followers:
            agent.fall_in(dt)

    # Meanders the commander as and calls the followers to follow
    def meander_lead(self, dt, v_max=.1, w_max=.05):
        self.meander(dt, v_max, w_max)
        for agent in self.followers:
            agent.fall_in(dt)

# Follower agent
class UnicycleFollower(UnicycleAgent):

    def __init__(self, commander, rel_pos, init_heading=None, init_heading_dot=None, init_vel=0., mass=1., moment_inertia=1.):
        # Initialize
        self.commander = commander
        self.target_rel_pos = np.array(rel_pos).flatten()
        init_pos = self.commander.pos + self.target_rel_pos
        if init_heading == None:
            init_heading = self.commander.heading
        if init_heading_dot == None:
            init_heading_dot = self.commander.heading_dot
        super().__init__(init_pos, init_vel, init_heading, init_heading_dot, mass, moment_inertia)
        # Initialize integrator
        # TODO

    # Relative error dotted with the desired direction
    def relative_pos_error_dotted(self):
        current_direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        relative_error = (self.target_rel_pos - self.pos + self.commander.pos)
        return np.dot(relative_error, current_direction)

    def relative_vel_error_dotted(self):
        current_direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        relative_error = (self.commander.vel - self.vel)
        return np.dot(relative_error, current_direction)

    def relative_heading_error(self):
        return self.wrap(self.commander.heading - self.heading)

    def relative_heading_dot_error(self):
        return self.commander.heading_dot - self.heading_dot

    # Calculate the forces using PID control
    def force_pid(self):
        P_vel = 20.
        D_vel = 8.
        P_rot = 50.
        D_rot = 20.
        F = P_vel * self.relative_pos_error_dotted() + D_vel * self.relative_vel_error_dotted()
        tau = P_rot * self.relative_heading_error() + D_rot * self.relative_heading_dot_error()
        return F, tau

    # Follow the commander
    def fall_in(self, dt):
        F, tau = self.force_pid()
        self.propagate(dt, F, tau)
