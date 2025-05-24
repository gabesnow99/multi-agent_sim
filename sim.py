import cv2

from point_agents import *

al = PointCommander()
al.generate_circle_of_followers(10)

dt = .01
tf = 15
n = int(tf / dt)

def cv_plot_agents(commander):

    red = (0, 0, 255)
    blue = (255, 0, 0)
    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4
    pixels_per_meter = 100
    width, height = int(pixels_per_meter * (x_max - x_min)), int(pixels_per_meter * (y_max - y_min))
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    def pos_to_pixels(x, y):
        x_pixel = int(pixels_per_meter * (x - x_min))
        y_pixel = int(pixels_per_meter * (y - y_min))
        return (x_pixel, y_pixel)

    cv2.circle(img, pos_to_pixels(commander.pos[0], commander.pos[1]), 6, red, -1)
    for agent in commander.followers:
        cv2.circle(img, pos_to_pixels(agent.pos[0], agent.pos[1]), 4, blue, -1)
        cv2.circle(img, pos_to_pixels(commander.pos[0] + agent.target_rel_pos[0], commander.pos[1] + agent.target_rel_pos[1]), 2, red, -1)
    cv2.imshow('Agents', img)

for frame in range(n):
    cv_plot_agents(al)

    if frame * dt < 1:
        al.meander(dt)
        for agent in al.followers:
            agent.meander(dt, max=.05)
    else:
        al.meander_lead(dt)

    if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
