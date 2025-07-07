import numpy as np
import random
from scipy.optimize import differential_evolution

from probability_visualization.probability_representations import Donut, MultiDonut


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
        # Used in range sensing and control
        self.estimated_pos = np.array([0, 0, 0])
        self.previous_estimated_pos = np.array([0, 0, 0])
        self.estimated_vel = np.array([0, 0, 0])
        # Used in particle filter localization
        self.particles = np.array([[]])
        # For debugging, otherwise comment out
        self.estimated_pos = np.copy(self.pos)
        self.estimated_vel = np.copy(self.vel)

    #################################### TRUE STATE MANIPULATION ####################################

    # Manually set the state
    def set_state(self, pos=None, vel=None):
        if pos != None:
            self.pos = pos
        if vel != None:
            self.vel = vel

    # Move forward a timestep
    def propagate(self, dt, F=[0., 0., 0.]):
        F = np.array(F).flatten()
        # Propagate pos and vel
        self.vel += dt * F / self.mass
        self.pos += dt * self.vel
        # Propagate estimated pos
        self.estimated_vel += dt * F / self.mass
        self.estimated_pos += dt * self.vel

    # Add a random amount to the velocity
    def meander(self, dt, max=.1):
        self.vel[0] += random.uniform(-max, max)
        self.vel[1] += random.uniform(-max, max)
        self.propagate(dt)

    ############################################ SENSING ############################################

    # When passed another PointAgent, calculates exact distance
    def range_to_agent(self, agent):
        return np.linalg.norm(self.pos - agent.pos)

    # When passed another PointAgent, return simulated range measurement
    def get_range_measurement(self, agent, stdev=.1):
        return self.range_to_agent(agent) + np.random.normal(loc=0, scale=stdev)

    # When passed another PointAgent, calculates exact bearing to that agent
    def bearing_to_agent(self, agent):
        return np.arctan2(agent.pos[1] - self.pos[1], agent.pos[0] - self.pos[0])

    ################################## DONUT SAMPLING LOCALIZATION ##################################

    # Localize using at least 3 range measurements to other agents
    def localize_donut_sampling(self, agents, dt, num_points=10000):

        if not isinstance(agents, list):
            print("Agents must be a list of Point Agents.")
            return
        if len(agents) < 3:
            print("At least 3 agents are required.")
            return

        stdev = .05
        dx, dy = .02, .02

        donuts = []
        for agent in agents:
            donut = Donut(self.get_range_measurement(agent, stdev), stdev, agent.estimated_pos[0], agent.estimated_pos[1], num_points=num_points)
            donuts.append(donut)
        md = MultiDonut(donuts, dx, dy)
        x_est, y_est, pos_prob = md.get_max_loc()

        self.previous_estimated_pos = self.estimated_pos
        self.estimated_pos = np.array([x_est, y_est, 0.])
        self.update_estimated_velocity(dt)

        return x_est, y_est

    ################################# PARTICLE FILTER LOCALIZATION #################################

    def localize_particle_filter(self, agents, dt, num_particles=1000, particle_grid=False):

        if not isinstance(agents, list):
            print("Agents must be a list of Point Agents.")
            return

        stdev = .05
        lamagia = .01
        ranges = np.array([self.get_range_measurement(agent, stdev=stdev) for agent in agents])

        # When starting with a fresh set of particles
        if len(self.particles[0]) < 2:
            # Calculate approximate useful region
            span = np.sum(ranges)
            x_min = self.estimated_pos[0] - span
            x_max = self.estimated_pos[0] + span
            y_min = self.estimated_pos[1] - span
            y_max = self.estimated_pos[1] + span
            self.generate_new_particles(num_particles, x_min, x_max, y_min, y_max)

        # Calculate the cost of each particle
        costs = []
        for particle in self.particles:
            dists = np.array([self.dist_point_to_agent(particle, agent) for agent in agents])
            # TODO: CHANGE TO MAHALANOBIS DISTANCE
            errs = np.abs(ranges - dists)
            c = np.prod(errs) + np.sum(errs) * lamagia
            costs.append(c)
        costs = np.array(costs)

        min_index = np.argmin(costs)
        x_est, y_est = self.particles[min_index, :2]

        # TODO: GOOD RESAMPLING
        best_particle = np.copy(self.particles[min_index, :]).flatten()
        np.random.shuffle(self.particles)
        twenty_index = int(.2 * len(self.particles))
        self.particles[:twenty_index, :] *= 0
        self.particles[:twenty_index, :] += best_particle

        self.chaosify_particles(.05)

        self.previous_estimated_pos = self.estimated_pos
        self.estimated_pos = np.array([x_est, y_est, 0.])
        self.update_estimated_velocity(dt)

        return x_est, y_est

    # Used to uniformly generate 2D particles within x and y bounds
    def generate_new_particles(self, n_particles, x_min, x_max, y_min, y_max):
        x = np.random.uniform(x_min, x_max, n_particles)
        y = np.random.uniform(y_min, y_max, n_particles)
        z = np.zeros(n_particles)
        self.particles = np.column_stack((x, y, z))
        return

    # Returns to the distance from the given point to the given agent
    def dist_point_to_agent(self, point, agent, use_estimated_pos=True):
        point = np.array(point).flatten()
        if use_estimated_pos:
            return np.linalg.norm(point - agent.estimated_pos)
        else:
            return np.linalg.norm(point - agent.pos)

    # Move the particles with the estimated velocity of the agent
    def propagate_particles(self, dt):
        self.particles += self.estimated_vel * dt
        return

    # Used to spread out the particles after duplicates have been created
    def chaosify_particles(self, stdev=.5):
        n_particles = np.shape(self.particles)[0]
        chaos_x = np.random.normal(0., stdev, n_particles)
        chaos_y = np.random.normal(0., stdev, n_particles)
        chaos = np.column_stack((chaos_x, chaos_y))
        self.particles[:, :2] += chaos
        return

    ######################################### OPTIMIZATION #########################################

    def optimized_localization(self, agents, dt):

        if not isinstance(agents, list):
            print("Agents must be a list of Point Agents.")
            return
        if len(agents) < 3:
            print("At least 3 agents are required.")
            return

        stdev = .05

        range_measurements = np.array([self.get_range_measurement(agent, stdev=stdev) for agent in agents])
        agents_locations = np.array([agent.estimated_pos[:2] for agent in agents])
        args = [range_measurements, agents_locations]

        span = np.sum(range_measurements)
        x_min = self.estimated_pos[0] - span
        x_max = self.estimated_pos[0] + span
        y_min = self.estimated_pos[1] - span
        y_max = self.estimated_pos[1] + span
        bounds = [(x_min, x_max), (y_min, y_max)]

        x_est, y_est = differential_evolution(self.sum_squared_cost, bounds, args=args).x
        self.estimated_pos = np.array([x_est, y_est, 0.])
        self.update_estimated_velocity(dt)

        return x_est, y_est

    def sum_squared_cost(self, xy, ranges, circles):
        dists = np.array([np.linalg.norm(np.array(circ) - np.array(xy)) for circ in circles])
        errs = np.abs(ranges - dists)
        cost = np.dot(errs, errs)
        return cost

    ###################################### GENERAL ESTIMATION ######################################

    def update_estimated_velocity(self, dt):
        self.estimated_vel = (self.estimated_pos - self.previous_estimated_pos) / dt

    ################################################################################################


# Commander agent
class PointCommander(PointAgent):

    def __init__(self, init_pos=[0., 0., 0.], init_vel=[0., 0., 0.], mass=1.):
        super().__init__(init_pos, init_vel, mass)
        self.followers = []

    # Used to generate a follower agent
    def create_follower(self, rel_pos, follower_id=None):
        self.followers.append(PointFollower(commander=self, rel_pos=rel_pos, follower_id=follower_id))

    # Generates n followers cluster in a circle of radius r about the commander
    def generate_circle_of_followers(self, n=4, r=1.):
        n = int(n)
        for ii in range(n):
            th = 2 * np.pi * ii / n
            self.create_follower(r * np.array([np.cos(th), np.sin(th), 0.]), follower_id=ii+1)

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

    def __init__(self, commander, rel_pos, P=15., I=0., D=3., init_vel=[0., 0., 0.], mass=1., follower_id=None):
        # Initialize
        self.commander = commander
        self.target_rel_pos = np.array(rel_pos).flatten()
        init_pos = self.commander.pos + self.target_rel_pos
        super().__init__(init_pos, init_vel, mass)
        self.follower_id = follower_id
        # Control relative to commander
        self.P = P
        self.I = I
        self.D = D
        # Initialize integrator
        self.integrator = 0.

    def distance_to_leader(self):
        return self.pos - self.commander.pos

    def estimated_distance_to_leader(self):
        return self.estimated_pos - self.commander.estimated_pos

    def relative_error(self):
        return self.target_rel_pos - self.distance_to_leader()

    def estimated_relative_error(self):
        return self.target_rel_pos - self.estimated_distance_to_leader()

    def relative_vel_error(self):
        return self.commander.vel - self.vel

    def estimated_relative_vel_error(self):
        return self.commander.estimated_vel - self.estimated_vel

    # Calculate the force using PID control
    def force_pid(self):
        err = self.estimated_relative_error()
        vel_err = self.estimated_relative_vel_error()
        self.integrator += err
        F = self.P * err + self.I * self.integrator + self.D * vel_err
        return F

    # Follow the commander
    def fall_in(self, dt):
        return
        F = self.force_pid()
        self.propagate(dt, F)
