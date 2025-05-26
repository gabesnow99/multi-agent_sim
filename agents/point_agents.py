import numpy as np
import random

# Super class of point agents
class PointAgent:

    '''
    self.pos (1x3 Array):   position (m) as [x, y, z]
    self.vel (1x3 Array):   velocity (m/s) as [x_dot, y_dot, z_dot]
    '''

    def __init__(self, init_pos=[0., 0., 0.], init_vel=[0., 0., 0.], mass=1.):
        self.pos = np.array(init_pos).flatten()
        self.vel = np.array(init_vel).flatten()
        self.mass = mass

    # Manually set the state
    def set_state(self, pos=None, vel=None):
        if pos:
            self.pos = pos
        if vel:
            self.vel = vel

    # Move forward a timestep
    def propagate(self, dt, F=[0., 0., 0.]):
        F = np.array(F).flatten()
        self.vel += dt * F / self.mass
        self.pos += dt * self.vel

    # Add a random amount to the velocity
    def meander(self, dt, max=.1):
        self.vel[0] += random.uniform(-max, max)
        self.vel[1] += random.uniform(-max, max)
        self.propagate(dt)

# Commander agent
class PointCommander(PointAgent):

    def __init__(self, init_pos=[0., 0., 0.], init_vel=[0., 0., 0.], mass=1.):
        super().__init__(init_pos, init_vel, mass)
        self.followers = []

    # Used to generate a follower agent
    def create_follower(self, rel_pos):
        self.followers.append(PointFollower(commander=self, rel_pos=rel_pos))

    # Generates n followers cluster in a circle of radius r about the commander
    def generate_circle_of_followers(self, n=4, r=1.):
        n = int(n)
        for ii in range(n):
            th = 2 * np.pi * ii / n
            self.create_follower(np.array([np.cos(th), np.sin(th), 0.]))

    # Marches the commander as well as all followers
    def forward_march(self, dt, F=[0., 0., 0.]):
        self.propagate(dt, F)
        for agent in self.followers:
            agent.fall_in(dt)

    # Meanders the commander as and calls the followers to follow
    def meander_lead(self, dt, rand_max=.1):
        self.meander(dt, rand_max)
        for agent in self.followers:
            agent.fall_in(dt)

# Follower agent
class PointFollower(PointAgent):

    def __init__(self, commander, rel_pos, P=15., I=0., D=3., init_vel=[0., 0., 0.], mass=1.):
        # Initialize
        self.commander = commander
        self.target_rel_pos = np.array(rel_pos).flatten()
        init_pos = self.commander.pos + self.target_rel_pos
        super().__init__(init_pos, init_vel, mass)
        # Control relative to commander
        self.P = P
        self.I = I
        self.D = D
        # Initialize integrator
        self.integrator = 0.

    def distance_to_leader(self):
        return self.pos - self.commander.pos

    def relative_error(self):
        return self.target_rel_pos - self.distance_to_leader()

    def relative_vel_error(self):
        return self.commander.vel - self.vel

    # Calculate the force using PID control
    def force_pid(self):
        err = self.relative_error()
        vel_err = self.relative_vel_error()
        self.integrator += err
        F = self.P * err + self.I * self.integrator + self.D * vel_err
        return F

    # Follow the commander
    def fall_in(self, dt):
        F = self.force_pid()
        self.propagate(dt, F)