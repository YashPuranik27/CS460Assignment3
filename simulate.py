import argparse
import numpy as np
import math
from controls import load_polygons, save_polygons, next_control

np.random.seed(43)


def actuation_model(planned_controls):
    exec_controls = [planned_controls[0]]

    for u in planned_controls[1:]:
        noise = np.array([np.random.normal(0, 0.075), np.random.normal(0, 0.2)])
        noise = np.where(u == 0, 0, noise)
        u_exec = u + noise
        u_exec = np.clip(u_exec, [-0.5, -0.9], [0.5, 0.9])
        exec_controls.append(u_exec)

    return np.array(exec_controls, dtype='object')


def odometry_model(executed_controls, z=True):
    sensed_controls = [executed_controls[0]]
    std_v, std_phi = (0.05, 0.1) if z else (0.1, 0.3)

    for u_exec in executed_controls[1:]:
        noise = np.array([np.random.normal(0, std_v), np.random.normal(0, std_phi)])
        noise = np.where(u_exec == 0, 0, noise)
        u_sensed = u_exec + noise
        sensed_controls.append(u_sensed)

    return np.array(sensed_controls, dtype='object')


def landmark_sensor(ground_truth_x, ground_truth_y, ground_truth_theta, landmarks):
    visible_landmarks_local = []

    for lx, ly in landmarks:
        dx, dy = lx - ground_truth_x, ly - ground_truth_y
        cos_theta, sin_theta = math.cos(-ground_truth_theta), math.sin(-ground_truth_theta)

        # Rotate and calculate distance and angle to the landmark
        rotated_x = dx * cos_theta - dy * sin_theta
        rotated_y = dx * sin_theta + dy * cos_theta
        distance = math.sqrt(rotated_x ** 2 + rotated_y ** 2)
        angle = math.atan2(rotated_y, rotated_x)

        # Add noise and append to visible landmarks
        noise = np.random.normal(0, 0.02)
        visible_landmarks_local.append([distance + noise, angle + noise])

    return np.array(visible_landmarks_local)


def get_gt(executed_controls):
    gt = [executed_controls[0]]
    for u in executed_controls[1:]:
        gt.append(next_control(gt[-1], u))
    return np.array(gt)


def get_readings(sensed_controls, gt_poses, landmarks):
    readings = [gt_poses[0]]
    for i in range(1, 201):
        x, y, theta = gt_poses[i]
        readings.extend([sensed_controls[i], landmark_sensor(x, y, theta, landmarks)])
    return np.array(readings, dtype='object')


def determine_z(reading_fname):
    if 'L' in reading_fname:
        return False
    else:
        return True


if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Simulate noisy robot motion and sensor readings')
    parser.add_argument('--plan', required=True, help='controls')
    parser.add_argument('--map', required=True, help='landmarks')
    parser.add_argument('--execution', required=True, help='gts')
    parser.add_argument('--sensing', required=True, help='readings')
    args = parser.parse_args()

    # Load and process data
    landmarks = load_polygons(args.map)
    executed_controls = actuation_model(load_polygons(args.plan))
    gt_poses = get_gt(executed_controls)
    sensed_controls = odometry_model(executed_controls, determine_z(args.sensing))
    readings = get_readings(sensed_controls, gt_poses, landmarks)

    # Save ground truths and readings (comment these two out to disable)
    save_polygons(gt_poses, args.execution)
    save_polygons(readings, args.sensing)
