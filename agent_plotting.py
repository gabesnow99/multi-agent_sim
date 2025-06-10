import cv2
import numpy as np

red = (0, 0, 255)
blue = (255, 0, 0)


def pos_to_pixels(x, y, x_min, x_max, y_min, y_max, pixels_per_meter):
    x_pixel = int(pixels_per_meter * (x - x_min))
    y_pixel = int(pixels_per_meter * (y_max - y))
    # y_pixel = int(pixels_per_meter * (y - y_min))
    return (x_pixel, y_pixel)

def cv_plot_unicycle_commander_and_followers(commander, dt, x_min=-4, x_max=4, y_min=-4, y_max=4, pixels_per_meter=100):

    width, height = int(pixels_per_meter * (x_max - x_min)), int(pixels_per_meter * (y_max - y_min))
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    def draw_unicycle(agent, color=red, tip_length=.6):
        # Calculate the locations of the end and tip of the arrow
        l = .05
        tip_vec = l * np.array([np.cos(agent.heading), np.sin(agent.heading)])
        tip_np = agent.pos + tip_vec * np.sign(agent.vel)
        end_np = agent.pos - tip_vec * np.sign(agent.vel)
        end = pos_to_pixels(end_np[0], end_np[1], x_min, x_max, y_min, y_max, pixels_per_meter)
        tip = pos_to_pixels(tip_np[0], tip_np[1], x_min, x_max, y_min, y_max, pixels_per_meter)
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

def cv_plot_commander_and_followers(commander, dt, x_min=-4, x_max=4, y_min=-4, y_max=4, pixels_per_meter=100):

    width, height = int(pixels_per_meter * (x_max - x_min)), int(pixels_per_meter * (y_max - y_min))
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Draw markers
    cv2.circle(img, pos_to_pixels(commander.pos[0], commander.pos[1], x_min, x_max, y_min, y_max, pixels_per_meter), 6, red, -1)
    for agent in commander.followers:
        cv2.circle(img, pos_to_pixels(agent.pos[0], agent.pos[1], x_min, x_max, y_min, y_max, pixels_per_meter), 4, blue, -1)
        cv2.circle(img, pos_to_pixels(commander.pos[0] + agent.target_rel_pos[0], commander.pos[1] + agent.target_rel_pos[1], x_min, x_max, y_min, y_max, pixels_per_meter), 2, red, -1)

    # Show and allow quitting
    cv2.imshow('Agents', img)
    if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
        quit()
