import cairo
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import networkx as nx


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


# Animates full path of robots
def animate_paths(robots, polygons):
    plt.clf()
    fig, ax = plt.subplots()

    # Lines for paths
    path_lines = [ax.plot([], [], label=f'Robot {i}')[0] for i, _ in enumerate(robots)]

    # Lines for orientation and velocity vectors
    orientation_lines = [ax.plot([], [], 'r-', linewidth=1)[0] for _ in robots]
    velocity_lines = [ax.plot([], [], 'g-', linewidth=1)[0] for _ in robots]

    # Plot polygons
    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    def init():
        return path_lines + orientation_lines + velocity_lines

    def update(frame):
        for i, robot in enumerate(robots):
            # Update path
            if frame < len(robot.position_history):
                x, y = zip(*robot.position_history[:frame + 1])
                path_lines[i].set_data(x, y)

            # Update orientation vector
            if frame < len(robot.orientation_history):
                ox, oy = robot.orientation_history[frame]
                pos_x, pos_y = robot.position_history[frame]
                orientation_lines[i].set_data([pos_x, pos_x + ox], [pos_y, pos_y + oy])

            # Update velocity vector
            if frame < len(robot.velocity_history):
                vx, vy = robot.velocity_history[frame]
                pos_x, pos_y = robot.position_history[frame]
                velocity_lines[i].set_data([pos_x, pos_x + vx], [pos_y, pos_y + vy])

        return path_lines + orientation_lines + velocity_lines

    anim = FuncAnimation(fig, update, frames=max(len(r.position_history) for r in robots),
                         init_func=init, blit=True)

    plt.legend()
    # plt.show()
    anim.save('paths.gif', fps=60)

    return anim


# Just converts continuous coordinates to occupancy grid coordinates
def continuous_to_grid(points, env_size, grid_size):
    points = np.array(points)
    env_size = np.array(env_size, dtype=float)
    grid_size = np.array(grid_size)

    grid_points = (points / env_size) * grid_size
    grid_points = np.clip(grid_points, [0, 0], grid_size - 1).astype(int)

    return grid_points


def animate_sim(robots, polygons):
    plt.clf()
    fig, ax = plt.subplots()

    # Scatter plot for robot positions
    scatters = [ax.scatter([], [], s=1, label=f'Robot {i}') for i, _ in enumerate(robots)]

    # Lines for orientation and velocity vectors
    orientation_lines = [ax.plot([], [], 'r-', linewidth=1)[0] for _ in robots]
    velocity_lines = [ax.plot([], [], 'g-', linewidth=1)[0] for _ in robots]

    # Plot polygons
    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    def init():
        return scatters + orientation_lines + velocity_lines

    def update(frame):
        for i, robot in enumerate(robots):
            # Update robot positions
            if frame < len(robot.position_history):
                x, y = robot.position_history[frame]
                scatters[i].set_offsets([x, y])

            # Update orientation vector
            if frame < len(robot.orientation_history):
                ox, oy = robot.orientation_history[frame]
                pos_x, pos_y = robot.position_history[frame]
                orientation_lines[i].set_data([pos_x, pos_x + ox * 5], [pos_y, pos_y + oy * 5])

            # Update velocity vector
            if frame < len(robot.velocity_history):
                vx, vy = robot.velocity_history[frame]
                pos_x, pos_y = robot.position_history[frame]
                velocity_lines[i].set_data([pos_x, pos_x + vx * 5], [pos_y, pos_y + vy * 5])

        return scatters + orientation_lines + velocity_lines

    anim = FuncAnimation(fig, update, frames=max(len(r.position_history) for r in robots),
                         init_func=init, blit=True)

    plt.legend()
    # plt.show()
    anim.save('sim_run.gif', fps=60)

    return anim


def update_grid(occupancy_grid, intersections, open_spaces, env_size, grid_size):
    grid_intersections = continuous_to_grid(intersections, env_size, grid_size)

    occupancy_grid[grid_intersections[:, 0], grid_intersections[:, 1]] = 0

    grid_open_spaces = continuous_to_grid(open_spaces, env_size, grid_size)

    occupancy_grid[grid_open_spaces[:, 0], grid_open_spaces[:, 1]] = 1

    return occupancy_grid


def generate_voronoi_graph(occupancy_grid):
    distance_map = distance_transform_edt(occupancy_grid >= 0)

    skeleton = skeletonize(distance_map > 0)

    graph = nx.Graph()
    rows, cols = skeleton.shape
    for r in range(rows):
        for c in range(cols):
            if skeleton[r, c]:
                graph.add_node((r, c))
                # Horizontal and vertical neighbors
                if r > 0 and skeleton[r - 1, c]:
                    graph.add_edge((r, c), (r - 1, c))
                if c > 0 and skeleton[r, c - 1]:
                    graph.add_edge((r, c), (r, c - 1))
                # Diagonal neighbors
                if r > 0 and c > 0 and skeleton[r - 1, c - 1]:
                    graph.add_edge((r, c), (r - 1, c - 1))
                if r > 0 and c < cols - 1 and skeleton[r - 1, c + 1]:
                    graph.add_edge((r, c), (r - 1, c + 1))
                if r < rows - 1 and c > 0 and skeleton[r + 1, c - 1]:
                    graph.add_edge((r, c), (r + 1, c - 1))
                if r < rows - 1 and c < cols - 1 and skeleton[r + 1, c + 1]:
                    graph.add_edge((r, c), (r + 1, c + 1))

    prune_nodes = []
    critical_nodes = []
    for node in graph.nodes:
        critical = False
        if graph.degree[node] == 2:
            neighbors = list(graph.neighbors(node))
            if any(graph.degree[n] == 3 for n in neighbors):
                # Check if it leads from known to unknown areas
                leads_to_unknown = False
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip the node itself
                        x, y = node[0] + dx, node[1] + dy
                        if 0 <= x < occupancy_grid.shape[0] and 0 <= y < occupancy_grid.shape[1]:
                            if occupancy_grid[x, y] == -1:  # Check for unknown area
                                leads_to_unknown = True
                                break
                    if leads_to_unknown:
                        critical = True
                        critical_nodes.append(node)
                        break

                # if not leads_to_unknown:
                #     prune_nodes.append(node)
        # if not critical:
        #     prune_nodes.append(node)

    # Remove pruned nodes
    # for node in prune_nodes:
    #     graph.remove_node(node)

    return graph, critical_nodes


def draw_occupancy(occupancy_grid, filename='occupancy.png'):
    occupancy_grid = np.flip(occupancy_grid, axis=1)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, occupancy_grid.shape[0] * 10, occupancy_grid.shape[1] * 10)
    ctx = cairo.Context(surface)

    # Draw the grid
    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            if occupancy_grid[x][y] == 0:
                ctx.rectangle(x * 10, y * 10, 10, 10)
                ctx.set_source_rgb(0, 0, 0)
            elif occupancy_grid[x][y] == 1:
                ctx.rectangle(x * 10, y * 10, 10, 10)
                ctx.set_source_rgb(1, 1, 1)
            else:
                ctx.rectangle(x * 10, y * 10, 10, 10)
                ctx.set_source_rgb(0.5, 0.5, 0.5)

            ctx.fill()

    surface.write_to_png(filename)


