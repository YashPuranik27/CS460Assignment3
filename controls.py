import numpy as np
from controls_driver import Car, draw_rotated_rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# global variable for final configuration and the value of pi
final_config = None
pi = np.pi


# Function to create a Matplotlib plot
def create_plot():
    fig, ax = plt.subplots(dpi=100)
    return ax


# Function to set up and show the plot
def show_scene(ax):
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()


# Function to save an array of polygons to a file
def save_polygons(polygons, filename):
    np.save(filename, arr=polygons, allow_pickle=True)


# Function to load polygons from a file
def load_polygons(filename):
    return np.load(filename, allow_pickle=True)


# Check if a point is within specified bounds
def within_bounds(pt):
    x, y, _ = pt
    if 0.2 <= x <= 1.8 and 0.2 <= y <= 1.8:
        return True
    else:
        return False


# Generate a random initial pose within the bounds
def genInitPose():
    return np.array([np.random.uniform(0.2, 1.8), np.random.uniform(0.2, 1.8), np.random.uniform(-pi, pi)])


# Compute the next pose based on current pose, control input, and time step
def next_control(q, u, dt=0.1):
    dq = np.zeros_like(q)
    dq[0] = u[0] * np.cos(q[2])
    dq[1] = u[0] * np.sin(q[2])
    dq[2] = u[1]

    next_q = q + dq * dt
    return next_q


# Generate a sequence of control inputs, ensuring each stays within bounds
def generate_controls(start):
    controls = []
    global final_config

    while len(controls) < 10:
        u = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.9, 0.9)])

        curr = start
        gen_configs = [next_control(curr, u) for _ in range(20)]

        if all(within_bounds(q) for q in gen_configs):
            controls.append(u)
            start = gen_configs[-1]

    final_config = start
    return controls


# Update function for the animation, applying controls to the car
def update_controls(frame, controls, car, visited, trace):
    x, y, _ = car.q

    if frame % 20 == 0:
        control_index = frame // 20
        v, phi = controls[control_index]
        car.u = np.array([v, phi])

    car.next_control()
    car.get_body()

    car.ax.add_patch(car.body)
    visited.append((x, y))

    trace.set_data(*zip(*visited))

    return [car.body, trace]


# Function to store control inputs and optionally visualize them
def store_controls(controls, initPose, X, Y, anim=False):
    landmarks = load_polygons(f'maps/landmark_{X}.npy')

    controls_to_store = [initPose] + [control for control in controls for _ in range(20)]
    controls_to_store = np.array(controls_to_store, dtype='object')

    save_polygons(controls_to_store, f'controls/controls_{X}_{Y}.npy')

    if anim:
        visualize(controls_to_store, landmarks, X, Y)


# Function to animate the car's movement and show the plot
def animate(landmarks, initPose, controls):
    # Create a new car object with the initial position
    diff_car = Car(ax=create_plot(), startConfig=initPose)

    # Initialize an empty list for tracking the visited positions
    visited = []

    car_trace, = plt.plot([], [], 'bo', label='Trace')

    plt.scatter(landmarks[:, 0], landmarks[:, 1])

    animate_var = FuncAnimation(diff_car.fig, update_controls, frames=200,
                                fargs=(controls, diff_car, visited, car_trace),
                                interval=100, blit=True, repeat=False)
    # Display the plot
    plt.show()


# Function to visualize the controls' effect on the car's trajectory
def visualize(controls, landmarks, X, Y):
    # Extract initial position and orientation
    q = controls[0]
    x, y, _ = q
    trajectory_x, trajectory_y = [x], [y]

    # Compute and store the trajectory
    for control in controls[1:]:
        q = next_control(q, control)
        x, y, _ = q
        trajectory_x.append(x)
        trajectory_y.append(y)

    # Plot landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1])

    # Plot the trajectory
    plt.plot(trajectory_x, trajectory_y, '--bo', label='Trace')

    # Draw the final position and orientation
    ax = plt.gca()
    draw_rotated_rectangle(ax, q[0:2], np.degrees(q[2] + np.pi / 2))

    # Display the scene
    show_scene(ax)


# Main execution block
if __name__ == '__main__':
    start = genInitPose()
    X = 0  # change X (change for each control)
    Y = 1  # change Y (change for each control)
    controls = generate_controls(start)
    landmarks = load_polygons('maps/landmark_0.npy')  # TODO: CHANGE landmark WHEN NEEDED
    # store_controls(controls, start, X, Y, anim=True)  # TODO: REMOVE comment "#" IF YOU NEED TO STORE CONTROLS
    animate(landmarks, start, controls)
