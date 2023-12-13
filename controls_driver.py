import matplotlib.pyplot as plt
from math import degrees, pi
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from itertools import combinations

# global lists to store collision data.
box_collisions = []
polygon_collisions = []


# This function calculates the bounding boxes for a list of polygons. A bounding box is the smallest rectangle that
# fully contains a polygon.
def bound_polygons(polygons):
    bbs = [np.array([polygon.min(axis=0), polygon.max(axis=0)]) for polygon in polygons]
    return bbs


# Determines if two bounding boxes overlap, indicating a potential collision between objects.
def check_box_collision(bbox1, bbox2):
    return not (bbox1[1][0] < bbox2[0][0] or bbox1[0][0] > bbox2[1][0] or
                bbox1[1][1] < bbox2[0][1] or bbox1[0][1] > bbox2[1][1])


# Applies the bounding box collision check to all combinations of given polygons.
def check_all_boxes(polygons):
    bb = bound_polygons(polygons)
    return [(polygons[i], polygons[j]) for i, j in combinations(range(len(bb)), 2) if check_box_collision(bb[i], bb[j])]


# Computes the edges of a polygon, used for collision detection.
def get_edges(polygons):
    return [polygons[i] - polygons[(i + 1) % len(polygons)] for i in range(len(polygons))]


# Finds the normal vectors for each edge of a polygon. These are used in the Separating Axis Theorem (SAT) for collision detection.
def get_normals(edges):
    normals = [np.array([-edge[1], edge[0]]) for edge in edges]
    return normals


# Projects the vertices of a polygon onto an axis, used in the SAT collision detection.
def project(vertices, axis):
    projections = [np.dot(axis, vertex) for vertex in vertices]
    return min(projections), max(projections)


# Implements the Separating Axis Theorem to check if two polygons intersect.
def SAT_Collides(polygon1, polygon2):
    normals = get_normals(get_edges(polygon1)) + get_normals(get_edges(polygon2))

    for normal in normals:
        min1, max1 = project(polygon1, normal)
        min2, max2 = project(polygon2, normal)
        if max1 < min2 or max2 < min1:
            return False
    return True


# Clears global collision lists and checks for collision between two polygons using both bounding box and SAT methods.
def collides(poly1: np.ndarray, poly2: np.ndarray):
    box_collisions.clear()
    polygon_collisions.clear()
    check_all_boxes([poly1, poly2])

    for p1, p2 in box_collisions:
        if SAT_Collides(p1, p2):
            return True
    return False


def get_coords(r1):
    return r1.get_corners()


# Ensures the car object remains within the defined boundaries of the environment.
def check_boundary(car):
    return all(0 <= x[0] <= 2 and 0 <= x[1] <= 2 for x in get_coords(car))


# Checks if the car collides with any of the obstacles in the environment.
def check_car(car, obstacles):
    return all(collides(polygon, get_coords(car)) for polygon in obstacles)


# Adds a polygon to the Matplotlib Axes object for visualization.
def add_polygon_to_scene(polygon, ax, fill):
    pol = plt.Polygon(polygon, closed=True, fill=fill, color='black', alpha=0.4)
    ax.add_patch(pol)


# Creates a rectangle patch representing a rigid body (like a car) with specified parameters.
def make_rigid_body(center, angle=0, opacity=0.5):
    return patches.Rectangle(
        (center[0] - 0.1, center[1] - 0.05),  # Adjusted for width / 2 and height / 2
        0.2,  # Width
        0.1,  # Height
        linewidth=1,
        angle=degrees(angle),
        rotation_point='center',
        edgecolor='r',
        facecolor='none',
        alpha=opacity
    )


# Car is a class that models a car-like object in the environment.
# It handles car state, motion, collision detection, and user input for control.
class Car:
    def __init__(self, ax: plt.Axes, startConfig=(1, 1, 0), u=[0, 0], w=0.2, h=0.1, dt=0.1, obs=[]):
        self.ax, self.fig = ax, ax.figure
        self.q, self.u, self.wid, self.ht, self.dt, self.obs = startConfig, u, w, h, dt, obs
        self.body, self.last_pos, self.continue_anim = None, [], True
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def set_obs(self, obstacles):
        self.obs = obstacles
        """
        Sets the obstacles for the car.
        :param obstacles: List of obstacle polygons.
        """

    def set_obs_plot(self):
        [add_polygon_to_scene(p, self.ax, False) for p in self.obs]
        """
        Adds the obstacle polygons to the matplotlib plot.
        """

    def dq(self):
        return np.array([self.u[0] * np.cos(self.q[2]),
                         self.u[0] * np.sin(self.q[2]),
                         self.u[1]]) * self.dt

    """
    Computes the change in position and orientation for the car based on its current state and control inputs.
    :return: Array representing change in position (dx, dy) and orientation (dtheta).
    """

    def new_position(self):
        self.q += self.dq()

    """
    Updates the car's position and orientation.
    """

    def next_control(self):
        self.last_pos.append(self.q)
        self.new_position()
        if collides_no_controller(self.body, self.obs):
            self.u, self.q = np.zeros_like(self.u), self.go_back()

    """
    Updates the car's state based on the next control input while checking for collisions.
    If a collision is detected, it reverses to the last safe position.
    """

    def go_back(self):
        return next((q for q in reversed(self.last_pos) if not collides_no_controller(self.get_body(), self.obs)),
                    np.zeros_like(self.q))

    """
    Reverses the car to the last safe position when a collision is detected.
    :return: The last safe position.
    """

    def get_body(self):
        x, y, theta = self.q
        self.body = patches.Rectangle((x - self.wid / 2, y - self.ht / 2), self.wid, self.ht, linewidth=1,
                                      edgecolor='black', facecolor='black')
        self.body.set_transform(Affine2D().rotate_deg_around(x, y, np.degrees(theta + pi / 2)) + self.ax.transData)

    """
    Updates the car's body for rendering based on its current position and orientation.
    """

    def on_key_press(self, event, v_min=-0.5, v_max=0.5, omega_min=-0.9, omega_max=0.9):
        if event.key == 'up':
            self.u[0] = np.clip(self.u[0] + 0.05, v_min, v_max)
        elif event.key == 'down':
            self.u[0] = np.clip(self.u[0] - 0.05, v_min, v_max)
        elif event.key in ['right', 'left']:
            self.u[1] = np.clip(self.u[1] + (0.1 if event.key == 'left' else -0.1), omega_min, omega_max)
        elif event.key == 'q':
            self.continue_anim = False

    """
    Handles keyboard events for controlling the car.
    :param event: The key press event.
    :param v_min: Minimum velocity.
    :param v_max: Maximum velocity.
    :param omega_min: Minimum angular velocity.
    :param omega_max: Maximum angular velocity.
    """

    def init_animation(self):
        self.get_body()
        self.ax.add_patch(self.body)
        return [self.body]

    """
    Initializes the animation by setting up the car's body.
    :return: A list containing the car's body for the animation.
    """

    def update_animation(self, frame):
        self.next()
        self.get_body()
        self.ax.add_patch(self.body)
        return [self.body]

    """
    Updates the animation for each frame.
    :param frame: The current frame number.
    :return: A list containing the updated car's body for the animation.
    """

    def start_animation(self):
        animation = FuncAnimation(self.fig, self.update_animation, init_func=self.init_animation, blit=True,
                                  repeat=False)
        plt.show()

    """
    Starts the animation loop.
    """


# Draws a rectangle at a given position, angle, and dimensions on an Axes object.
def draw_rotated_rectangle(ax, center, angle_degrees, width=0.2, height=0.1, color='black'):
    x, y = center
    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=1, edgecolor=color,
                             facecolor=color)
    rect.set_transform(Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData)
    ax.add_patch(rect)


# Checks if a car body collides with obstacles or boundaries without considering its controller.
def collides_no_controller(car_body, obstacles):
    return car_body and not (check_car(car_body, obstacles) and check_boundary(car_body))
