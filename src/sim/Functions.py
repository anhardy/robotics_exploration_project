from collections import deque

import cairo
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import networkx as nx


# Rotate a 2D vector by a given angle.
def rotate_vector(vector, angles):
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    rotation_matrix = np.array([
        [cos_angles, -sin_angles],
        [sin_angles, cos_angles]
    ])

    # Adjusting the shapes for broadcasting
    rotation_matrix = np.transpose(rotation_matrix, (2, 0, 1))
    vector = vector.reshape((2, 1))

    # Broadcasting the dot product over the angles
    return np.dot(rotation_matrix, vector).reshape(-1, 2)


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

    if len(open_spaces) > 0:
        grid_open_spaces = continuous_to_grid(open_spaces, env_size, grid_size)

        occupancy_grid[grid_open_spaces[:, 0], grid_open_spaces[:, 1]] = 1

    if len(intersections) > 0:
        grid_intersections = continuous_to_grid(intersections, env_size, grid_size)

        occupancy_grid[grid_intersections[:, 0], grid_intersections[:, 1]] = 0

    return occupancy_grid


def is_accessible_to_frontier(node, occupancy_grid, max_distance=5):
    # Perform a BFS to find if there is accessible frontier space within distance
    visited = set()
    queue = deque([(node, 0)])  # (node, distance)

    while queue:
        (r, c), distance = queue.popleft()
        if distance > max_distance:
            break  # Limit the search to distance

        # Check for frontier space
        if occupancy_grid[r][c] == -1:
            return True

        # Add neighbors to the queue
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (len(occupancy_grid) > nr >= 0 != occupancy_grid[nr][nc] and 0 <= nc < len(occupancy_grid[0])
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), distance + 1))

    return False


def find_critical_nodes(graph, occupancy_grid):
    critical_nodes = []
    segments = []

    # Identify leaf nodes
    leaf_nodes = [node for node in graph.nodes if graph.degree[node] == 1]

    for leaf in leaf_nodes:
        current = leaf
        path = [current]
        frontier_found = False

        while True:
            # Find the next node in the path
            neighbors = list(graph.neighbors(current))
            next_node = None
            for neighbor in neighbors:
                if neighbor not in path:
                    next_node = neighbor
                    break

            if next_node is None:
                break  # No next node, end of path

            path.append(next_node)
            current = next_node

            # Check for frontier adjacency if not already found
            if not frontier_found:
                frontier_found = is_accessible_to_frontier(current, occupancy_grid)

            # Check if current node is a critical node
            if frontier_found and graph.degree[current] == 2 and any(
                    graph.degree[neighbor] == 3 for neighbor in graph.neighbors(current)):
                critical_nodes.append(current)
                segments.append(path)
                break

    return critical_nodes, segments


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        if node not in self.parent:
            self.parent[node] = node
            return node
        if self.parent[node] == node:
            return node
        self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1
            return True
        return False


def generate_voronoi_graph(occupancy_grid):
    distance_map = distance_transform_edt(occupancy_grid >= 0)
    skeleton = skeletonize(distance_map > 0)
    graph = nx.Graph()
    uf = UnionFind()

    rows, cols = skeleton.shape
    nodes = set()
    edges = set()

    # Collect nodes and potential edges
    for r in range(rows):
        for c in range(cols):
            if skeleton[r, c]:
                nodes.add((r, c))
                for dr, dc in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and skeleton[nr, nc]:
                        edge = ((r, c), (nr, nc))
                        edges.add(edge)

    # Add nodes and edges to the graph
    graph.add_nodes_from(nodes)
    for edge in edges:
        if uf.union(*edge):
            graph.add_edge(*edge)

    critical_nodes, segments = find_critical_nodes(graph, occupancy_grid)

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