import matplotlib.pyplot as plt
from math import degrees, pi
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from itertools import combinations


box_collisions = []
polygon_collisions = []

def bound_polygons(polygons):
    bbs = [np.array([polygon.min(axis=0), polygon.max(axis=0)]) for polygon in polygons]
    return bbs


def check_box_collision(bbox1, bbox2):
    return not (bbox1[1][0] < bbox2[0][0] or bbox1[0][0] > bbox2[1][0] or
                bbox1[1][1] < bbox2[0][1] or bbox1[0][1] > bbox2[1][1])


def check_all_boxes(polygons):
    bb = bound_polygons(polygons)
    return [(polygons[i], polygons[j]) for i, j in combinations(range(len(bb)), 2) if check_box_collision(bb[i], bb[j])]


def get_edges(polygons):
    return [polygons[i] - polygons[(i + 1) % len(polygons)] for i in range(len(polygons))]


def get_normals(edges):
    normals = [np.array([-edge[1], edge[0]]) for edge in edges]
    return normals


def project(vertices, axis):
    projections = [np.dot(axis, vertex) for vertex in vertices]
    return min(projections), max(projections)


def SAT_Collides(polygon1, polygon2):
    normals = get_normals(get_edges(polygon1)) + get_normals(get_edges(polygon2))

    for normal in normals:
        min1, max1 = project(polygon1, normal)
        min2, max2 = project(polygon2, normal)
        if max1 < min2 or max2 < min1:
            return False
    return True


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


def check_boundary(car):
    return all(0 <= x[0] <= 2 and 0 <= x[1] <= 2 for x in get_coords(car))


def check_car(car, obstacles):
    return all(collides(polygon, get_coords(car)) for polygon in obstacles)


def add_polygon_to_scene(polygon, ax, fill):
    pol = plt.Polygon(polygon, closed=True, fill=fill, color='black', alpha=0.4)
    ax.add_patch(pol)


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

    def set_obs_plot(self):
        [add_polygon_to_scene(p, self.ax, False) for p in self.obs]

    def dq(self):
        return np.array([self.u[0] * np.cos(self.q[2]),
                         self.u[0] * np.sin(self.q[2]),
                         self.u[1]]) * self.dt

    def new_position(self):
        self.q += self.dq()

    def next_control(self):
        self.last_pos.append(self.q)
        self.new_position()
        if collides_no_controller(self.body, self.obs):
            self.u, self.q = np.zeros_like(self.u), self.go_back()

    def go_back(self):
        return next((q for q in reversed(self.last_pos) if not collides_no_controller(self.get_body(), self.obs)),
                    np.zeros_like(self.q))

    def get_body(self):
        x, y, theta = self.q
        self.body = patches.Rectangle((x - self.wid / 2, y - self.ht / 2), self.wid, self.ht, linewidth=1,
                                      edgecolor='black', facecolor='black')
        self.body.set_transform(Affine2D().rotate_deg_around(x, y, np.degrees(theta + pi / 2)) + self.ax.transData)

    def on_key_press(self, event, v_min=-0.5, v_max=0.5, omega_min=-0.9, omega_max=0.9):
        if event.key == 'up':
            self.u[0] = np.clip(self.u[0] + 0.05, v_min, v_max)
        elif event.key == 'down':
            self.u[0] = np.clip(self.u[0] - 0.05, v_min, v_max)
        elif event.key in ['right', 'left']:
            self.u[1] = np.clip(self.u[1] + (0.1 if event.key == 'left' else -0.1), omega_min, omega_max)
        elif event.key == 'q':
            self.continue_anim = False

    def init_animation(self):
        self.get_body()
        self.ax.add_patch(self.body)
        return [self.body]

    def update_animation(self, frame):
        self.next()
        self.get_body()
        self.ax.add_patch(self.body)
        return [self.body]

    def start_animation(self):
        animation = FuncAnimation(self.fig, self.update_animation, init_func=self.init_animation, blit=True,
                                  repeat=False)
        plt.show()


def draw_rotated_rectangle(ax, center, angle_degrees, width=0.2, height=0.1, color='black'):
    x, y = center
    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=1, edgecolor=color, facecolor=color)
    rect.set_transform(Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData)
    ax.add_patch(rect)


def collides_no_controller(car_body, obstacles):
    return car_body and not (check_car(car_body, obstacles) and check_boundary(car_body))

