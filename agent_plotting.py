import cv2
import numpy as np

red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)

global pixels_per_meter
global x_min, x_max, y_min, y_max
pixels_per_meter = 100
x_min, x_max, y_min, y_max = -4, 4, -4, 4

def set_pixels_per_meter(ppm):
    global pixels_per_meter
    pixels_per_meter = ppm

def set_xy_bounds(x_min_, x_max_, y_min_, y_max_):
    global x_min, x_max, y_min, y_max
    x_min, x_max, y_min, y_max = x_min_, x_max_, y_min_, y_max_

def pos_to_pixels(x, y):
    x_pixel = int(pixels_per_meter * (x - x_min))
    y_pixel = int(pixels_per_meter * (y_max - y))
    # y_pixel = int(pixels_per_meter * (y - y_min))
    return (x_pixel, y_pixel)

def range_to_pixels(r):
    return int(pixels_per_meter * r)

def add_point(img, x, y):
    cv2.circle(img, pos_to_pixels(x, y), 4, green, -1)

def cv_plot_unicycle_commander_and_followers(commander, dt):

    width, height = int(pixels_per_meter * (x_max - x_min)), int(pixels_per_meter * (y_max - y_min))
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    def draw_unicycle(agent, color=red, tip_length=.6):
        # Calculate the locations of the end and tip of the arrow
        l = .05
        tip_vec = l * np.array([np.cos(agent.heading), np.sin(agent.heading)])
        tip_np = agent.pos + tip_vec * np.sign(agent.vel)
        end_np = agent.pos - tip_vec * np.sign(agent.vel)
        end = pos_to_pixels(end_np[0], end_np[1])
        tip = pos_to_pixels(tip_np[0], tip_np[1])
        thickness = 2
        # Draw arrow
        cv2.arrowedLine(img, end, tip, color, thickness, tipLength=tip_length)

    draw_unicycle(commander)
    for agent in commander.followers:
        draw_unicycle(agent, color=blue)

    # Show and allow quitting
    cv2.imshow('Agents', img)
    if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
        quit()

def cv_plot_commander_and_followers(commander, dt, draw_target_loc=False, range_rings_indices=np.array([[]]), estimate=np.array([[]])):

    width, height = int(pixels_per_meter * (x_max - x_min)), int(pixels_per_meter * (y_max - y_min))
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Draw markers beginning with the commander
    cv2.circle(img, pos_to_pixels(commander.pos[0], commander.pos[1]), 6, red, -1)
    # Commander range circles
    if np.shape(range_rings_indices)[0] > 1:
        indices_to_range = range_rings_indices[:, 0]
        for ii, torf in enumerate(indices_to_range):
            if torf:
                if ii == 0:
                    target_agent = commander
                else:
                    target_agent = commander.followers[ii - 1]
                cv2.circle(img, pos_to_pixels(commander.pos[0], commander.pos[1]), range_to_pixels(commander.range_to_agent(target_agent)), red, 0)
    # Draw followers
    for ii, agent in enumerate(commander.followers):
        # Plot follower agent
        cv2.circle(img, pos_to_pixels(agent.pos[0], agent.pos[1]), 4, blue, -1)
        # Plot follower agent target location
        if draw_target_loc:
            cv2.circle(img, pos_to_pixels(commander.pos[0] + agent.target_rel_pos[0], commander.pos[1] + agent.target_rel_pos[1]), 2, red, -1)
        # Plot follower agent range rings
        if np.shape(range_rings_indices)[0] > 1:
            indices_to_range = range_rings_indices[:, ii+1]
            for jj, torf in enumerate(indices_to_range):
                if torf:
                    if jj == 0:
                        target_agent = commander
                    else:
                        target_agent = commander.followers[jj - 1]
                    cv2.circle(img, pos_to_pixels(agent.pos[0], agent.pos[1]), range_to_pixels(agent.range_to_agent(target_agent)), red, 0)

    # Plot the estimate of the
    if len(estimate[0]) == 2:
        for p in estimate:
            add_point(img, p[0], p[1])

    # Show and allow quitting
    cv2.imshow('Agents', img)
    if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
        quit()

    return img
