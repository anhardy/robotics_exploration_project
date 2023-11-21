import numpy as np

# from src.environment.RandomEnv import SimEnv
from src.sim.helpers import rotate_vector, line_intersection, is_between, plot_detection, angle_between


class Robot:
    def __init__(self, position, angle_range=np.pi / 3, num_vectors=5, scale=50):
        self.position = np.array(position)
        self.velocity = np.array([0, 0])
        self.acceleration = np.array([0, 0])
        self.orientation = np.random.randint(-360, 360, size=(2,))  # np.array([1, 0])
        self.angle_range = angle_range
        self.num_vectors = num_vectors
        self.scale = scale
        self.perception_cone = self.get_perception_cone()

    def update_velocity(self):
        self.velocity += self.acceleration
        # Only update orientation if velocity is nonzero
        if np.any(self.velocity != 0):
            magnitude = np.linalg.norm(self.velocity)
            if magnitude != 0:
                self.orientation = self.velocity / magnitude
        self.perception_cone = self.get_perception_cone()

    # Generate vectors in a cone around the orientation vector.
    def get_perception_cone(self):
        vectors = []
        orientation_angle = np.arctan2(self.orientation[1], self.orientation[0])
        start_angle = orientation_angle - self.angle_range / 2
        end_angle = orientation_angle + self.angle_range / 2
        angles = np.linspace(start_angle, end_angle, self.num_vectors)

        for angle in angles:
            vector = rotate_vector(self.orientation, angle - orientation_angle)
            normalized_vector = vector / np.linalg.norm(vector)
            scaled_vector = normalized_vector * self.scale
            vectors.append(scaled_vector)

        return vectors

    # Detects closest intersections with polygons in the perception cone.
    def detect(self, polygons):
        closest_intersections = []

        for vector in self.perception_cone:
            closest_point = None
            min_distance = np.inf

            for polygon in polygons:
                for i in range(len(polygon)):
                    line1 = [self.position, self.position + vector]
                    line2 = [polygon[i], polygon[(i + 1) % len(polygon)]]

                    intersection = line_intersection(line1, line2)
                    if intersection and is_between(line2[0], line2[1], intersection):
                        to_intersection = np.array(intersection) - self.position
                        if (np.linalg.norm(to_intersection) < np.linalg.norm(vector) and angle_between(vector,
                                                                                                       to_intersection)
                                <= np.pi / 2):
                            distance = np.linalg.norm(to_intersection)
                            if distance < min_distance:
                                min_distance = distance
                                closest_point = intersection

            if closest_point:
                closest_intersections.append(closest_point)

        return closest_intersections


# env = SimEnv(width=250, height=250, min_room_size=25, max_room_size=50, min_rooms=20, max_rooms=20, hallway_width=5,
#              n_robots=5, r_radius=2, rand_connections=0)
# env.print_grid()
# env.draw_env('env.png')
# # obstacles = env.get_obstacles()
# env.scale_grid(1000, 1000)
# polygons = env.convert_to_poly()
# test_rob = Robot(env.starting_points[0])
# test_detect = test_rob.detect(polygons)
# plot_detection(test_rob.position, test_rob.perception_cone, polygons, test_detect)

# print('test')
