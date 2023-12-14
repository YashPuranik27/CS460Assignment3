import argparse
import numpy as np
import math
from numpy.linalg import norm
from numpy.random import randn, uniform
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from controls import create_plot, load_polygons, save_polygons
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from controls_driver import Car

#python  ./particle_filter.py --map maps/landmark_0.npy  --sensing readings/readings_0_1_H.npy --num_particles 100

np.random.seed(43)

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))



def resample_from_index_old(particles, weights, indexes):

    print("............. resample from index", particles)
    #particles[:] = particles[indexes]
    #weights.resize(len(particles))
    #weights.fill(1.0 / len(weights))


    new_particles = []
    # Normalize weights
    norm_weights = weights / np.sum(weights)

    # Draw representative sample
    num_particles = len(particles)
    indices = [np.random.choice(np.arange(0, num_particles),
                                    p=norm_weights) for i in range(num_particles)]

    print("------------- indices", indices, weights, norm_weights)

    for i in indices:
        new_particles.append(particles[i])
    assert num_particles == len(new_particles)

    # Update particles
    return new_particles





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
def move(particles, u, car):
    #move according to control input u (velocity, theta)

    particles[:, 0] += u[0]*np.cos(particles[:, 2])
    particles[:, 1] += u[0]*np.sin(particles[:, 2])
    particles[:, 2] += u[1]*car.dt



def Gaussian(mu, sigma=1.0, x=0.0):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return math.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / math.sqrt(2.0 * math.pi * (sigma ** 2))


def measurement_prob(x, y, landmarks):
    # calculates how likely a measurement should be

    prob = 1.0

    for i in range(len(landmarks)):
        #dist = math.sqrt(((x - landmarks[i][0]) ** 2) + ((y - landmarks[i][1]) ** 2))

        dist = math.sqrt(49)
        prob *= Gaussian(dist)

    return prob




def UpdateWeights(weights, particles, landmarks, ground_readings):

    for i in range(len(particles)):

        noise  = 0.2
        # determin the weight based on how much the particle and the robot's measurements are alike
        w = 1
        # estimate particle to landmark distance base on observation model

        my_measurements = landmark_sensor(particles[i][0], particles[i][1], particles[i][2], landmarks)
        for j in range(len(landmarks)):
            my_distance = my_measurements[j][0]
            ground_distance = ground_readings[j][0]

            print("xxxxxxxxxxxxxx      ",my_distance,  ground_distance)
            w *= Gaussian(mu=my_distance, sigma=noise, x=ground_distance)

        weights[i] = w + 1.e-300  # avoid round-off to zero


#Normalizing the weights so they sum to one turns them into a probability distribution.
#The particles those that are closest to the robot
#will generally have a higher weight than ones far from the robot.
def update_particles(particles, weights, z, R, landmarks, gournd_readings):

    print("--landmarks   ", landmarks)
    print("--gournd_readings   ", gournd_readings)


    #UpdateWeights(weights, particles, landmarks, gournd_readings)

    std_err = .1
    zs = []

    for i, gournd_reading in enumerate(gournd_readings):    # distance from robot to each landmark
        zs.append(gournd_reading[0])

    print("--zs   ", zs)


    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        #print("........... distance ", distance)
        weights *= scipy.stats.norm(distance, std_err).pdf(zs[i])

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)  # normalize



    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

    print("---------------------- WEIGHTS ", weights)



def neff(weights):
    return 1. / np.sum(np.square(weights))


def stratified_resample(weights):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = (np.random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

#
#particle filter
#
def run_pf_old(landmarks, N):
    iters=18
    sensor_std_err=.1,
    do_plot=True
    plot_particles=False,
    xlim=(0, 2)
    ylim=(0, 2)
    initial_x=None

    #landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    NL = len(landmarks)
    print("NL= ", NL)

    plt.figure()

    # create particles and weights
    #if initial_x is not None:
    #    particles = create_gaussian_particles(mean=initial_x, std=(5, 5, np.pi / 4), N=N)
    #else:
    particles = get_uniform_particles((0, 2), (0, 2), (0, 6.28), N)
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
        #move(particles, u=(0.00, 1.414), std=(.2, .05))

        # incorporate measurements
        update_particles(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks)

        # resample if too few effective particles
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            print("...........indexes ", indexes)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1 / N)


        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1],color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    print("   estimation array", xs)
    xs = np.array(xs)
    # plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()



def load_landmark_readings(readings):
    return np.array(readings[2::2])


def load_sensed_controls(readings):
    return np.array(readings[1::2])


def get_body(ax, center, angle_degrees, width=0.2, height=0.1, color='black'):
    x, y = center
    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height,
                             linewidth=1, edgecolor=color, facecolor='none')
    rect.set_transform(Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData)
    return rect


"""def get_landmark_pos(measure, pos):
    return np.array([np.array([dist * np.cos(angle), dist * np.sin(angle)]) + pos[:2] for dist, angle in measure])"""


def estimate_landmark_position(robot_x, robot_y, robot_theta, measurements):
    #cos_theta, sin_theta = math.cos(robot_theta), math.sin(robot_theta)

    cos_theta = 1
    sin_theta = 1
    return np.array([
        [
            robot_x + distance * math.cos(angle) * cos_theta - distance * math.sin(angle) * sin_theta,
            robot_y + distance * math.cos(angle) * sin_theta + distance * math.sin(angle) * cos_theta
        ]
        for distance, angle in measurements
    ])


def resample_particles(particles, weights):
    # ------------------------------------------------------------------------
    # calculate probability and "window" of particle getting picked in "resampling stage"
    # IE get normalized cumulative weights
    # ------------------------------------------------------------------------
    normalized_prob = np.array(weights) / sum(weights)
    cumulative_prob = []
    current = 0.
    number_of_particles = len(particles)

    for i in range(number_of_particles):
        cumulative_prob.append(current)
        current += normalized_prob[i]

    # ------------------------------------------------------------------------
    # get the new set of particles by "resampling"
    # ------------------------------------------------------------------------
    resampled_particles = []

    for i in range(number_of_particles):

        current = np.random.random()
        j = number_of_particles - 1

        while cumulative_prob[j] > current:
            j -= 1

        resampled_particles = particles[j]

    return resampled_particles




def do_pf(particles, u, xs, pos, car, weights, landmark_readings):

    #-----------
    # MOVE particles based on noisy odometry measurement
    #-----------
    sensor_std_err = .1
    #print("======= u ", u, praticles)
    move(particles, u, car)
    print("after move ", particles, weights)



    #-----------
    # WEIGHT particles
    #-----------

    # distance from robot to each landmark
    N = len(particles)
    zs = (norm(landmarks - pos, axis=1))
    print("before update", particles)
    update_particles(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks, gournd_readings=landmark_readings)
    print("after update", particles, weights)



    #-----------
    # RESAMPLE
    #-----------
    #praticles = resample_particles(particles, weights)

    # resample if too few effective particles
    if neff(weights) < N / 2:
        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)
        assert np.allclose(weights, 1 / N)



    # -----------
    # EsTIMATE
    # -----------
    mu, var = estimate(particles, weights)

    xs.append(mu)



def update(frame, sensed, landmark_readings, car, visited1, landmarks, landmark_plot, trace1, visited2, particles, xs, initPose):

    N = len(particles)

    # Update car1 state and plot
    car.u = sensed[frame]

    car.next_control()
    car.get_body()
    car.ax.add_patch(car.body)
    x, y, theta = car.q
    visited1.append((x, y))
    trace1.set_data(*zip(*visited1))


    weights = np.ones(N) / N

    print("************************* initPos ", initPose, "car x ", x, "car y ", y)
    landmark_pos = estimate_landmark_position(initPose[0], initPose[1], 1, landmark_readings[frame])
    #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^ landmarks", landmark_positions)

    #landmark_plot.set_offsets(landmark_positions)


    # Update ground truth trace
    #visited2.append(poses[frame][:2])
    #trace2.set_data(*zip(*visited2))


    #clear_particles = plt.scatter(particles[:, 0], particles[:, 1], color='white', marker=',', s=1)
    pos = (x, y)
    do_pf(particles, car.u, xs, pos, car, weights, landmark_readings[frame])
    new_particles = plt.scatter(particles[:, 0], particles[:, 1], color='k', marker=',', s=1)


    return [car.body, trace1, landmark_plot, new_particles]

def show_scene(ax):
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()





def show_animation(landmarks, readings, xs, nParticles, estFile):

    initPose = readings[0]

    # create particles and weights
    # for known init position
    particles = create_gaussian_particles(mean=initPose, std=(5, 5, np.pi / 4), N=nParticles)
    # for unkonwn init position
    #particles = create_uniform_particles((0, 2), (0, 2), (0, 6.28), nParticles)

    # run_pf(landmarks, nParticles)




    # Extract sensed controls and landmark measurements
    controls = load_sensed_controls(readings)
    landmark_observation  =   load_landmark_readings(readings)
    #print("!!!!!!!!!!!!!!!!!!!!! landmark_observation ", landmark_observation[0], "\ncontrols = ", controls)

    dead_reckon_car = Car(ax=create_plot(), startConfig=initPose)

    #particles = get_uniform_particles((0, 2), (0, 2), (0, 5), nParticles)
    # print(particles)
    #run_pf(landmarks, nParticles)

    visited1, visited2 = [], []
    car_trace, = plt.plot([], [], 'ro', label='Car Trace')
    #gt_trace, = plt.plot([], [], '', label='', color='white')

    numFrame = 200

    #for frame in range(200):

    #for frame in range(10):
        #plt.clf()
        #ax = plt.gca()
    plt.xlim(0, 2)
    plt.ylim(0, 2)

    landmark_plot = plt.scatter(landmarks[:, 0], landmarks[:, 1], color='red', marker='x')

    plt.scatter(particles[:, 0], particles[:, 1], color='k', marker=',', s=1, label='Particles')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], alpha=1, color='blue', label='Landmarks')

        #update(frame, controls, landmark_readings, dead_reckon_car, visited1, landmarks, landmark_plot, car_trace, visited2, particles, xs, initPose)

    plt.pause(0.05)
    ani = FuncAnimation(dead_reckon_car.fig, update, frames=numFrame,
                        fargs=(controls, landmark_observation, dead_reckon_car, visited1, landmarks, landmark_plot, car_trace, visited2, particles, xs, initPose),
                        interval=200, blit=True, repeat=False)



    plt.legend()
    plt.show()


    print("   estimation array", xs)
    save_polygons(xs, estFile)
    plt.close("all")


if __name__ == '__main__':

    # Setup argument parser

    parser = argparse.ArgumentParser(description='Particle Filter for localization')
    parser.add_argument('--map', required=True, help='landmarks')
    #parser.add_argument('--execution', required=True, help='gts')
    parser.add_argument('--sensing', required=True, help='readings')
    parser.add_argument('--num_particles', required=True, help='nParticles')
    parser.add_argument('--estimates', required=True, help='estimates')


    args = parser.parse_args()
    nParticles = int(args.num_particles)


    # Load data
    landmarks = load_polygons(args.map)
    readings = load_polygons(args.sensing)
    estFile = args.estimates
    #gt = load_polygons(args.execution)



    # Show animation

    xs=[]
    show_animation(landmarks, readings, xs, nParticles, estFile)




