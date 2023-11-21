from src.environment.RandomEnv import SimEnv
from src.sim.behaviors.steer_behavior import steer_behavior
from src.sim.behaviors.avoidance_behavior import avoidance_behavior
from src.sim.functions import plot_robot_paths, animate_paths
from src.sim.robot import Robot
from src.sim.robot_controller import RobotController

robots = []

env = SimEnv(width=250, height=250, min_room_size=25, max_room_size=50, min_rooms=20, max_rooms=20, hallway_width=5,
             n_robots=5, r_radius=2, rand_connections=0)
env.scale_grid(1000, 1000)
polygons = env.convert_to_poly()
controller = RobotController(1, 500, steer_behavior=steer_behavior, avoid_behavior=avoidance_behavior)

for robot in env.starting_points:
    robots.append(Robot(robot, max_vel=2.5))


for _ in range(100):
    for robot in robots:
        robot.acceleration = controller.calculate_acceleration(robot, [500, 500], polygons)
        robot.update_velocity()
        robot.update_position()

plot_robot_paths(robots, polygons)

animate_paths(robots, polygons)

