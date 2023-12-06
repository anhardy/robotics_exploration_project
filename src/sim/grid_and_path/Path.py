import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
import networkx as nx

from src.environment.PathGraph import update_graph


# Just converts continuous coordinates to occupancy grid coordinates
def continuous_to_grid(points, env_size, grid_size):
    points = np.array(points)

    grid_points = (points / env_size) * grid_size
    grid_points = np.clip(grid_points, [0, 0], grid_size - 1).astype(int)

    return grid_points


def update_grid(occupancy_grid, intersections, open_spaces, env_size, grid_size, path_graph, pad_range=3):
    if len(open_spaces) > 0:
        grid_open_spaces = continuous_to_grid(open_spaces, env_size, grid_size)

        update_mask = occupancy_grid[grid_open_spaces[:, 0], grid_open_spaces[:, 1]] != 0
        valid_updates = grid_open_spaces[update_mask]
        occupancy_grid[valid_updates[:, 0], valid_updates[:, 1]] = 1

    if len(intersections) > 0:
        grid_intersections = continuous_to_grid(intersections, env_size, grid_size)

        pad_grid = np.zeros_like(occupancy_grid, dtype=bool)
        for dx in range(-pad_range, pad_range + 1):
            for dy in range(-pad_range, pad_range + 1):
                shifted_intersections = grid_intersections + [dx, dy]

                # Filter out points that are outside the grid boundaries
                valid_points = (shifted_intersections[:, 0] >= 0) & (shifted_intersections[:, 0] < grid_size[0]) & \
                               (shifted_intersections[:, 1] >= 0) & (shifted_intersections[:, 1] < grid_size[1])
                valid_shifted_intersections = shifted_intersections[valid_points]

                # Update the pad_grid
                pad_grid[valid_shifted_intersections[:, 0], valid_shifted_intersections[:, 1]] = True

        # Update the occupancy grid with the padded intersections
        occupancy_grid[pad_grid] = 0
        changed_cells = np.where(pad_grid)

        changed_cells = list(zip(changed_cells[0], changed_cells[1]))
        path_graph = update_graph(path_graph, occupancy_grid, changed_cells)

    frontier_grid = update_frontier_grid(occupancy_grid)

    return occupancy_grid, frontier_grid, path_graph


def update_frontier_grid(occupancy_grid):
    frontier_grid = np.zeros_like(occupancy_grid)

    open_spaces = occupancy_grid == 1
    unexplored_spaces = occupancy_grid == -1

    # Identify adjacent cells of open spaces
    # Shift the open spaces grid in all four directions and check for unexplored spaces
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in shifts:
        shifted_open_spaces = np.roll(open_spaces, shift=(dx, dy), axis=(0, 1))

        # Mark frontier cells
        frontier_grid[np.logical_and(shifted_open_spaces, unexplored_spaces)] = 1

    return frontier_grid


def find_critical_nodes(graph, occupancy_grid):
    critical_nodes = set([])
    segments = []

    # Identify leaf nodes
    leaf_nodes = [node for node in graph.nodes if graph.degree[node] == 1]
    all_paths = []

    for leaf in leaf_nodes:
        current = leaf
        path = [current]
        # frontier_found = False

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
            # if not frontier_found:
            #     frontier_found = is_accessible_to_frontier(current, occupancy_grid)

            # Check if current node is a critical node
            if graph.degree[current] == 2 and any(
                    graph.degree[neighbor] == 3 for neighbor in graph.neighbors(current)):
                segments.append(path)
                critical_nodes.add(current)
                break
        all_paths.append(path)
    if len(segments) == 0:
        segments = all_paths

    return critical_nodes, segments, leaf_nodes


# Just used to eliminate loops
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


def assign_frontier_targets_to_segments(frontier_grid, segments):
    """Finds unexplored space adjacent to observed open spaces and labels it as frontier space, then assigns this
    frontier space to the closest segment."""
    node_coordinates = []
    segment_indices = []
    for i, segment in enumerate(segments):
        for coord in segment:
            node_coordinates.append(coord)
            segment_indices.append(i)

    node_coordinates = np.array(node_coordinates)
    segment_indices = np.array(segment_indices)
    if len(node_coordinates) == 0:
        print('what')

    frontier_cells = np.argwhere(frontier_grid == 1)

    # Using broadcasting to create a distance matrix
    distances = np.sqrt(((frontier_cells[:, np.newaxis, :] - node_coordinates) ** 2).sum(axis=2))

    closest_node_indices = np.argmin(distances, axis=1)
    assigned_segments = segment_indices[closest_node_indices]

    target_coordinates = [[] for _ in segments]

    for cell, segment_index in zip(frontier_cells, assigned_segments):
        target_coordinates[segment_index].append(cell.tolist())

    return target_coordinates


# Needed for nx A*
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def find_unobstructed_node(robot_pos, nodes, occupancy_grid):
    robot_x, robot_y = robot_pos
    unobstructed_nodes = []
    distances = []

    for node in nodes:
        node_x, node_y = node

        x = np.linspace(robot_x, node_x, num=100)
        y = np.linspace(robot_y, node_y, num=100)
        sampled_grid_values = occupancy_grid[np.round(x).astype(int), np.round(y).astype(int)]

        if np.all(sampled_grid_values) != 0:
            distance = np.linalg.norm(robot_pos - node)
            unobstructed_nodes.append(node)
            distances.append(distance)

    # Find the closest unobstructed node
    if unobstructed_nodes:
        closest_node = unobstructed_nodes[np.argmin(distances)]
        return closest_node

    return None


def find_nearest_node(graph, position):
    nodes = np.array(graph.nodes())
    distances = np.linalg.norm(nodes - position, axis=1)
    nearest_node = tuple(nodes[np.argmin(distances)])
    return nearest_node


def assign_paths(graph, robots, segments, frontier_targets, nodes, path_graph, occupancy_grid):
    """
    Calculates paths to segments and assigns these paths to robots
    using the Hungarian method.
    """
    costs = {}
    paths_dict = {}
    nodes_array = np.array(list(nodes))

    for robot in robots:
        robot_costs = {}
        robot_paths = {}

        distances_to_voronoi = np.linalg.norm(nodes_array - robot.position, axis=1)
        nearest_voronoi_node = tuple(nodes_array[np.argmin(distances_to_voronoi)])
        position = tuple(np.round(robot.position).astype(int))
        try:
            # First path segment: from robot to nearest node on Voronoi graph
            path_to_voronoi = nx.astar_path(path_graph, position, nearest_voronoi_node,
                                            heuristic=heuristic)
        except nx.NodeNotFound:
            # If robot's position is not in the global graph, find the nearest node
            nearest_node_in_global_graph = find_nearest_node(path_graph, position)
            path_to_voronoi = [position, nearest_node_in_global_graph]

        for i, segment in enumerate(segments):
            try:

                if len(frontier_targets[i]) == 0:
                    continue

                # Nearest frontier cell to the end point
                frontier_cells = np.array(frontier_targets[i])

                distances = np.linalg.norm(frontier_cells - robot.position, axis=1)

                min_index = np.argmin(distances)

                # Get the closest frontier cell
                closest_frontier_cell = tuple(frontier_cells[min_index])

                distances = np.linalg.norm(segment - np.array(closest_frontier_cell), axis=1)

                min_index = np.argmin(distances)

                nearest_node_in_segment = tuple(segment[min_index])

                # Second path segment: on Voronoi graph
                try:
                    path_on_voronoi = nx.astar_path(graph, nearest_voronoi_node, nearest_node_in_segment, heuristic=heuristic)
                except Exception as e:
                    path_on_voronoi = nx.astar_path(path_graph, nearest_voronoi_node, nearest_node_in_segment,
                                                    heuristic=heuristic)
                    # print(e)
                    # nx.draw_networkx_nodes(graph, pos={n: n for n in graph.nodes}, node_color='blue', node_size=1)
                    # nx.draw_networkx_edges(graph, pos={n: n for n in graph.nodes}, edge_color='red')
                    #
                    # plt.savefig('skeleton_crash.png')
                    # plt.show()
                    # exit(0)

                # Third path segment: from Voronoi graph to end point
                # try:
                path_to_endpoint = nx.astar_path(path_graph, nearest_node_in_segment, closest_frontier_cell,
                                                 heuristic=heuristic)
                # except Exception as e:
                #     print(e)
                #     nx.draw_networkx_nodes(graph, pos={n: n for n in graph.nodes}, node_color='blue', node_size=1)
                #     nx.draw_networkx_edges(graph, pos={n: n for n in graph.nodes}, edge_color='red')
                #
                #     plt.savefig('skeleton_crash.png')
                #     plt.show()
                #     exit(0)

                # Combine paths
                complete_path = path_to_voronoi + path_on_voronoi[1:] + path_to_endpoint[1:]  # Avoid duplicating nodes

                # Calculate path cost and store paths
                path_cost = len(complete_path)
                robot_costs[i] = path_cost
                robot_paths[i] = complete_path

            except Exception as e:
                continue

        costs[robot] = robot_costs
        paths_dict[robot] = robot_paths

    cost_matrix = np.full((len(robots), len(segments)), 99999)
    for i, robot in enumerate(robots):
        for j in range(len(segments)):
            cost_matrix[i, j] = costs[robot].get(j, 99999)

    # Initial assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = {robot: None for robot in robots}
    for i, j in zip(row_ind, col_ind):
        assignment[robots[i]] = paths_dict[robots[i]][j] if j < len(segments) else None

    # Identify unassigned robots
    unassigned_robots = [robot for robot, path in assignment.items() if path is None]

    prev_length = len(unassigned_robots)
    # Iterative reassignment
    while unassigned_robots:
        # Recompute cost matrix for unassigned robots
        new_cost_matrix = cost_matrix[[robots.index(robot) for robot in unassigned_robots], :]
        # new_cost_matrix[:, col_ind] = 99999  # Set costs of assigned segments to a high value

        # Find new assignments for unassigned robots
        row_ind, col_ind_new = linear_sum_assignment(new_cost_matrix)
        for i, j in zip(row_ind, col_ind_new):
            robot = unassigned_robots[i]
            assignment[robot] = paths_dict[robot][j] if j < len(segments) else None

        # Update unassigned robots
        unassigned_robots = [robot for robot, path in assignment.items() if path is None]

        if len(unassigned_robots) == prev_length:
            break

        # Update the previous length for the next iteration
        prev_length = len(unassigned_robots)
        unassigned_robots = [robot for robot, path in assignment.items() if path is None]

    for robot in robots:
        robot.path = np.array(assignment[robot]) if assignment[robot] is not None else None
        if robot.path is not None:
            robot.path_history.append(np.copy(robot.path))
            robot.path_len = len(robot.path)


def generate_voronoi_graph(occupancy_grid):
    skeleton = skeletonize(occupancy_grid > 0)
    graph = nx.Graph()
    # uf = UnionFind()

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
        # if uf.union(*edge):  # and not line_close_to_wall(edge, occupancy_grid):
        graph.add_edge(*edge)

    critical_nodes, segments, leaf_nodes = find_critical_nodes(graph, occupancy_grid)

    return graph, critical_nodes, segments, nodes, leaf_nodes
