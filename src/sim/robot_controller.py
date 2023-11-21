import numpy as np


class RobotController:
    def __init__(self, steer_weight, avoid_weight, steer_behavior, avoid_behavior=None):
        self.steer_weight = steer_weight
        self.avoid_weight = avoid_weight
        self.steer_behavior = steer_behavior
        self.avoid_behavior = avoid_behavior

    def calculate_acceleration(self, robot, target, obstacles):
        steer_vector = self.steer_behavior(robot, target) * self.steer_weight
        if self.avoid_behavior is not None:
            avoid_vector = self.avoid_behavior(robot, obstacles) * self.avoid_weight
        else:
            avoid_vector = np.array([0, 0])

        acceleration = steer_vector + avoid_vector
        return acceleration
