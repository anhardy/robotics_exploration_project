import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# Rotate a 2D vector by a given angle.
def rotate_vector(vector, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(rotation_matrix, vector)


def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# Finds the intersection point of two lines
def line_intersection(line1, line2):
    xdiff = np.array([line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]])
    ydiff = np.array([line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines don't intersect

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# Check if point c is on line segment ab
def is_between(a, b, c):
    cross_product = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    if abs(cross_product) > 1e-7:
        return False

    dot_product = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
    if dot_product < 0:
        return False

    squared_length = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    if dot_product > squared_length:
        return False

    return True


def plot_detection(robot_position, perception_cone, polygons, intersections):
    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    # Plot the detection vectors
    for vector in perception_cone:
        end_point = robot_position + vector
        plt.arrow(robot_position[0], robot_position[1],
                  end_point[0] - robot_position[0], end_point[1] - robot_position[1],
                  head_width=0.05, head_length=0.1, fc='r', ec='r')

    # Plot the intersection points
    for point in intersections:
        plt.plot(point[0], point[1], 'go')

    # Mark the robot's position
    plt.plot(robot_position[0], robot_position[1], 'ko')

    plt.title('Robot Perception')
    plt.axis('equal')
    plt.show()


def plot_robot_paths(robots, polygons):
    plt.figure(figsize=(10, 10))

    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    for robot in robots:
        x_positions = [position[0] for position in robot.position_history]
        y_positions = [position[1] for position in robot.position_history]

        plt.plot(x_positions, y_positions, marker='o', markersize=1)

    plt.title('Robot Positions')
    plt.legend([f'Robot {i + 1}' for i in range(len(robots))])
    plt.savefig('sim_output.png')


def animate_paths(robots, polygons):
    plt.clf()
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=f'Robot {i}')[0] for i, _ in enumerate(robots)]
    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    def init():
        return lines

    def update(frame):
        for line, robot in zip(lines, robots):
            if frame < len(robot.position_history):
                x, y = zip(*robot.position_history[:frame + 1])
                line.set_data(x, y)
        return lines

    anim = FuncAnimation(fig, update, frames=max(len(r.position_history) for r in robots),
                         init_func=init, blit=True)

    plt.legend()
    # plt.show()
    anim.save('paths.gif', fps=60)

    return anim


