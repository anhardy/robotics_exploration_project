import numpy as np


def steer_behavior(robot, target):
    desired_direction = target - robot.position
    # desired_velocity = desired_direction / np.linalg.norm(
    #     desired_direction)  # * robot.max_vel

    steering_force = desired_direction - robot.velocity
    return steering_force
