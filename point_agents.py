import numpy as np


# Super class of point agents
class PointAgent:

    def __init__(self, init_pos=[0., 0., 0.], init_vel=[0., 0., 0.], mass=1.):
        self.pos = np.array(init_pos).flatten()
        self.vel = np.array(init_vel).flatten()
        self.mass = mass

    def set_state(self, pos=None, vel=None):
        if pos:
            self.pos = pos
        if vel:
            self.vel = vel

    def propagate(self, dt, F=[0., 0., 0.]):
        F = np.array(F).flatten()
        self.vel += dt * F / self.mass
        self.pos += dt * self.vel

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


class PointFollower(PointAgent):

    def __init__(self, commander, rel_pos, P=15., I=0., D=1., init_vel=[0., 0., 0.], mass=1.):
        # Initialize
        self.commander = commander
        self.target_rel_pos = np.array(rel_pos).flatten()
        init_pos = self.commander.pos + self.target_rel_pos
        super().__init__(init_pos, init_vel, mass)
        # Control relative to commander
        self.P = P
        self.I = I
        self.D = D
        self.integrator = 0.

    def distance_to_leader(self):
        return self.pos - self.commander.pos

    def relative_error(self):
        return self.target_rel_pos - self.distance_to_leader()

    def fall_in(self, dt):
        err = self.relative_error()
        self.integrator += err
        F = self.P * err + self.I * self.integrator
        self.propagate(dt, F)