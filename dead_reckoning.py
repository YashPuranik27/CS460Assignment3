import argparse
import numpy as np
from controls_driver import Car
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from controls import create_plot, load_polygons
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import math


# load_landmark_readings and load_sensed_controls extract the landmark and sensed controls input data from the .npy
# files
def load_landmark_readings(readings):
    return np.array(readings[2::2])


def load_sensed_controls(readings):
    return np.array(readings[1::2])


# get_body creates the car that is positioned and rotated based on the parameters given
def get_body(ax, center, angle_degrees, width=0.2, height=0.1, color='black'):
    x, y = center
    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height,
                             linewidth=1, edgecolor=color, facecolor='none')
    rect.set_transform(Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData)
    return rect


"""def get_landmark_pos(measure, pos):
    return np.array([np.array([dist * np.cos(angle), dist * np.sin(angle)]) + pos[:2] for dist, angle in measure])"""


# estimate_landmark_position basically calculates the estimated positions of the landmarks based on the position of
# the robot and sensor measurements
def estimate_landmark_position(robot_x, robot_y, robot_theta, measurements):
    cos_theta, sin_theta = math.cos(robot_theta), math.sin(robot_theta)
    return np.array([
        [
            robot_x + distance * math.cos(angle) * cos_theta - distance * math.sin(angle) * sin_theta,
            robot_y + distance * math.cos(angle) * sin_theta + distance * math.sin(angle) * cos_theta
        ]
        for distance, angle in measurements
    ])


# update() literally just updates the state of the car and how it is represented on the plot. It also updates
# the landmark positions based on the data from the "sensors", then it updates the cars trace and the ground truth (gt)
def update(frame, sensed, sensors, car1, visited1, landmarks, trace1, visited2, trace2, poses):
    # Update car1 state and plot
    car1.u = sensed[frame]
    car1.next_control()
    car1.get_body()
    car1.ax.add_patch(car1.body)
    x, y, theta = car1.q
    visited1.append((x, y))
    trace1.set_data(*zip(*visited1))

    # Update landmarks based on estimated positions
    landmark_positions = estimate_landmark_position(poses[frame][0], poses[frame][1], poses[frame][2], sensors[frame])
    landmarks.set_offsets(landmark_positions)

    # Update ground truth trace
    visited2.append(poses[frame][:2])
    trace2.set_data(*zip(*visited2))

    return [car1.body, trace1, trace2, landmarks]


# show_animation uses FuncAnimation that repeatedly calls update() in order to animate the car
def show_animation(landmarks, initPose, controls, sensors, poses):
    dead_reckon_car = Car(ax=create_plot(), startConfig=initPose)
    visited1, visited2 = [], []
    car_trace, = plt.plot([], [], 'ro', label='Car Trace')
    gt_trace, = plt.plot([], [], 'bo', label='GT Trace')
    landmark_plot = plt.scatter(landmarks[:, 0], landmarks[:, 1], color='red', marker='x')

    plt.scatter(landmarks[:, 0], landmarks[:, 1], label='Landmarks')
    ani = FuncAnimation(dead_reckon_car.fig, update, frames=200,
                        fargs=(
                            controls, sensors, dead_reckon_car, visited1, landmark_plot, car_trace, visited2, gt_trace,
                            poses),
                        interval=100, blit=True, repeat=False)
    plt.legend()
    plt.show()


# main method just gets the args via argparse, loads the data files that are parsed via argparse, then extracts the
# readings and sensing data as well as call show_animation to display the animation
if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description='Visualize actual robot position and estimated position')
    parser.add_argument('--map', required=True, help='landmarks')
    parser.add_argument('--execution', required=True, help='gts')
    parser.add_argument('--sensing', required=True, help='readings')
    args = parser.parse_args()

    # Load data
    landmarks = load_polygons(args.map)
    gt = load_polygons(args.execution)
    readings = load_polygons(args.sensing)

    # Extract sensed controls and landmark measurements
    sensed_controls = load_sensed_controls(readings)
    landmark_readings = load_landmark_readings(readings)

    # Show animation
    show_animation(landmarks, gt[0], sensed_controls, landmark_readings, gt)
