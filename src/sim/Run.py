import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

from src.environment.RandomEnv import SimEnv
from src.sim.behaviors.steer_behavior import steer_behavior
from src.sim.behaviors.avoidance_behavior import avoidance_behavior
from src.sim.Functions import plot_robot_paths, animate_paths, animate_sim, update_grid, generate_voronoi_graph, \
    draw_occupancy, draw_frontier_grid, assign_frontier_targets_to_segments, assign_paths, plot_segment_with_frontier
from src.sim.Robot import Robot
from src.sim.RobotController import RobotController
from scipy.spatial import Voronoi, voronoi_plot_2d
import cProfile
import pstats
from io import StringIO

robots = []

env = SimEnv(width=250, height=250, min_room_size=20, max_room_size=50, min_rooms=15, max_rooms=15, hallway_width=8,
             n_robots=5, r_radius=2, rand_connections=1)
env.scale_grid(750, 750)

controller = RobotController(1, 2, steer_behavior=steer_behavior, avoid_behavior=avoidance_behavior)

for robot in env.starting_points:
    robots.append(Robot(robot, max_vel=2.5, num_vectors=40, angle_range=np.pi / 3, perception_range=30, avoid_range=2,
                        arrival_range=20))

grid_height = 750
grid_width = 750
occupancy_grid = np.full((grid_height, grid_width), -1)
poly_arr = env.polygon_arr
polygons = env.polygon

pr = cProfile.Profile()
pr.enable()

# Predefining to reduce computation
env_size = np.array((env.width, env.height), dtype=float)
grid_size = np.array((grid_width, grid_height))
grid_history = [np.copy(occupancy_grid)]
for i in range(250):
    all_intersections = []
    all_open_spaces = []
    for robot in robots:
        nearby_lines = env.quadtree.query_range(robot.position, robot.perception_range)
        intersections, open_space = robot.detect(nearby_lines)
        all_intersections += intersections
        all_open_spaces += open_space
        robot.acceleration, robot.is_avoiding = (
            controller.calculate_acceleration(robot, robot.path, intersections))
        robot.update_velocity()
        robot.update_position()

    occupancy_grid, frontier_grid = update_grid(occupancy_grid, all_intersections, all_open_spaces, env_size,
                                                grid_size)
    grid_history.append(np.copy(occupancy_grid))

    # TODO possibly better way of determining when to do an update of the graph + reassignment of segments/frontiers
    if i % 5 == 0:
        graph, critical_points, segments, nodes, leaf_nodes = generate_voronoi_graph(occupancy_grid)
        frontier_targets = assign_frontier_targets_to_segments(frontier_grid, segments)

        good_segments = []
        good_targets = []
        for j, segment in enumerate(segments):
            if len(frontier_targets[j]) > 0:
                good_segments.append(segment)
                good_targets.append(frontier_targets[j])

        # TODO only do perception on robots currently assigned to a task. Need to initialize robots differently.
        #   ALL robots should still be considered when updating the graph in case assignments change
        assign_paths(graph, robots, good_segments, good_targets, nodes)
        # for robot in robots:
        #     closest_node =
    # plt.clf()
    # plt.cla()
    # plt.imshow(np.transpose(occupancy_grid), cmap='gray', alpha=0.5)
    # for polygon in polygons:
    #     polygon_with_closure = polygon + [polygon[0]]
    #     x, y = zip(*polygon_with_closure)
    #     plt.plot(x, y, 'b-')

    # nx.draw_networkx_nodes(graph, pos={n: n for n in graph.nodes}, node_color='blue', node_size=50)
    # # Edges
    # nx.draw_networkx_edges(graph, pos={n: n for n in graph.nodes}, edge_color='red')

    # plt.grid(False)  # Turn off the grid if not needed
    # plt.show()

pr.disable()

s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')

ps.print_stats()

print(s.getvalue())


draw_occupancy(occupancy_grid)
draw_frontier_grid(frontier_grid, occupancy_grid)
plot_segment_with_frontier(0, segments, frontier_targets)
plt.clf()
plt.cla()
# plt.imshow(np.transpose(occupancy_grid), cmap='gray', alpha=0.5)
for polygon in polygons:
    polygon_with_closure = polygon + [polygon[0]]
    x, y = zip(*polygon_with_closure)
    plt.plot(x, y, 'b-')

nx.draw_networkx_nodes(graph, pos={n: n for n in graph.nodes}, node_color='blue', node_size=1)
nx.draw_networkx_edges(graph, pos={n: n for n in graph.nodes}, edge_color='red')
for point in critical_points:
    plt.scatter(*point, color='green', s=10)

plt.grid(False)  # Turn off the grid if not needed
plt.savefig('skeleton.png')
plt.show()

plot_robot_paths(robots, polygons)

# animate_paths(robots, polygons)
animate_sim(robots, polygons, grid_history)
