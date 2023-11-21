import numpy as np


def avoidance_behavior(robot, polygons):
    avoidance_vector = np.array([0.0, 0.0])

    intersections = robot.detect(polygons)

    for point in intersections:
        # Vector from intersection to robot
        to_robot = robot.position - np.array(point)
        distance = np.linalg.norm(to_robot)

        # Normalize and inversely weigh by distance
        if 0 < distance <= robot.avoid_range:
            to_robot_normalized = to_robot / distance
            avoidance_vector += to_robot_normalized / distance

    if np.linalg.norm(avoidance_vector) > 0:
        avoidance_vector = avoidance_vector / np.linalg.norm(avoidance_vector)  # * max_avoidance_speed

    return avoidance_vector
