import argparse
import numpy as np
import math
from numpy.linalg import norm
from numpy.random import randn, uniform
import scipy.stats
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample


np.random.seed(43)

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))

def get_uniform_particles(xRange, yRange, headingRange, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(xRange[0], xRange[1], size=N)
    particles[:, 1] = uniform(yRange[0], yRange[1], size=N)
    particles[:, 2] = uniform(headingRange[0], headingRange[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def load_polygons(filename):
    return np.load(filename, allow_pickle=True)

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



#move each particle by control
def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist


#Normalizing the weights so they sum to one turns them into a probability distribution.
#The particles those that are closest to the robot
#will generally have a higher weight than ones far from the robot.
def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

def neff(weights):
    return 1. / np.sum(np.square(weights))


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

#
#particle filter
#
def run_pf(landmarks, N):
    iters=18
    sensor_std_err=.1,
    do_plot=True
    plot_particles=False,
    xlim=(0, 20)
    ylim=(0, 20)
    initial_x=None

    #landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    NL = len(landmarks)
    print("NL= ", NL)

    plt.figure()

    # create particles and weights
    #if initial_x is not None:
    #    particles = create_gaussian_particles(mean=initial_x, std=(5, 5, np.pi / 4), N=N)
    #else:
    particles = get_uniform_particles((0, 20), (0, 20), (0, 6.28), N)
    weights = np.ones(N) / N

    if plot_particles:
        alpha = .20
    if N > 5000:
        alpha *= np.sqrt(5000) / np.sqrt(N)
        plt.scatter(particles[:, 0], particles[:, 1],alpha=alpha, color='g')

    xs = []
    robot_pos = np.array([0., 0.])
    for x in range(iters):
        robot_pos += (1, 1)

        # distance from robot to each landmark
        zs = (norm(landmarks - robot_pos, axis=1) +(randn(NL) * sensor_std_err))

        # move diagonally forward to (x+1, x+1)
        predict(particles, u=(0.00, 1.414), std=(.2, .05))

        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err,
                   landmarks=landmarks)

        # resample if too few effective particles
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1 / N)
        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1],color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    xs = np.array(xs)
    # plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()


if __name__ == '__main__':
    # Setup argument parser
    print("start...................")
    parser = argparse.ArgumentParser(description='Simulate noisy robot motion and sensor readings')
    parser.add_argument('--map', required=True, help='landmarks')
    parser.add_argument('--sensing', required=True, help='readings')
    parser.add_argument('--num_particles', required=True, help='nParticles')
    #parser.add_argument('--estimates', required=True, help='readings')
    args = parser.parse_args()


    # Load and process data
    landmarks = load_polygons(args.map)
    print(landmarks)

    readings = load_polygons(args.sensing)
    print(readings)

    nParticles=int(args.num_particles)
    print(nParticles)

    #particles = get_uniform_particles((0, 1), (0, 1), (0, 5), 100)
    #print(particles)
    run_pf(landmarks, nParticles)

    #executed_controls = actuation_model(load_polygons(args.plan))
    #gt_poses = get_gt(executed_controls)
    #sensed_controls = odometry_model(executed_controls, determine_z(args.sensing))
    #readings = get_readings(sensed_controls, gt_poses, landmarks)

    # Save ground truths and readings (comment these two out to disable)
    #save_polygons(gt_poses, args.execution)
    #save_polygons(readings, args.sensing)


    #generate robot pose estimaties at each time step


    #initialize all of theparticles at the same pose ( use initial (ground truth) robot pose


    #genearte an animation that visualizes the particles at each iteration of the algorithm
    #it should visualize at least the (x,y)location of the robot
    #it will be nice to draw a bar to indicates the robot's orientation
    # plot all particles on the map do you cna see how particles move over consecutive iterations

    # the particle filter algorithm does need the map of landmarks as input in order to compute the likelihood of landmark observations from different robot pose estimates

    #store each iteration of the algorithm the mean particle estimate for the (x, y , theta) coordinates of the robot, the format is similar to the ground truth



