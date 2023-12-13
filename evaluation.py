import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from controls import load_polygons, create_plot
from controls_driver import Car


# This is a function to update the animation at each frame.
def update_animation(frame, car, gts, ests, gt_trace, estimate_trace, gt_visit, estimate_visit):
    car.body.remove() if car.body else None  # here we remove the cars body if it exists
    # getting the ground truth and estimated positions
    gt_position, estimated_position = gts[frame + 1], ests[frame + 1]
    car.set_q(*gt_position)  # Update the cars position
    car.ax.add_patch(car.get_body())  # Draw the new position of the car

    gt_visit.append(gt_position[:2])  # Append the ground truth position to the visit list
    estimate_visit.append(estimated_position[:2])  # Append the estimated position to the visit list

    gt_trace.set_data(*zip(*gt_visit))  # Update the ground truth trace
    estimate_trace.set_data(*zip(*estimate_visit))  # Update the estimate trace

    return [car.body, gt_trace, estimate_trace]  # Return the elements to be updated in the animation


# Function to generate plots comparing ground truth and estimates.
def generate_plots(gts, estimates):
    assert np.allclose(gts[0], estimates[0])  # Assert that the first elements of gts and estimates are close

    def angular_distance(angle1, angle2):
        # Calculate and return the angular distance between two angles
        return np.abs(np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2)))

    fig, (ax1, ax2) = plt.subplots(2, 1)  # Create a figure with two subplots
    fig.suptitle('GT vs estimated poses error')  # Set the title of the figure

    # Plot Euclidean distance between gts and estimates on the first subplot
    ax1.plot(np.arange(201), np.linalg.norm(gts[:, 0:2] - estimates[:, 0:2], axis=1), '.-')
    ax1.set_ylabel('Translational error relative to the ground truth pose (Euclidean Distance)')

    # Plot angular distance on the second subplot
    ax2.plot(np.arange(201), angular_distance(gts[:, 2], estimates[:, 2]), '.-')
    ax2.set_ylabel('Rotational error relative to the ground truth pose (Angular Distance')
    ax2.set_xlabel('time step')

    plt.show()  # Display the plots


# I got the plot from here:
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplot.html#sphx-glr-gallery-subplots-axes-and-figures-subplot-py

# Function to show the animation.
def play_animation(landmarks, gts, estimates):
    def initialize_plots():
        make_plot = create_plot()  # Create and initialize the plot
        make_plot.scatter(landmarks[:, 0], landmarks[:, 1])  # Scatter plot the landmarks
        return make_plot

    ax = initialize_plots()  # Initialize the plot
    # Plot the initial gts and estimated traces
    gt_trace, = ax.plot(gts[0][0], gts[0][1], 'bo', label='GT Trace')
    estimated_trace, = ax.plot(estimates[0][0], estimates[0][1], 'ko', label='Estimate Trace')

    # this creates an animation with 200 frames using the update() function
    FuncAnimation(ax.figure, update_animation, frames=200,
                  fargs=(Car(ax, startConfig=gts[0]), gts, estimates, gt_trace, estimated_trace),
                  blit=True, repeat=False)
    plt.show()  # shows the animation


# main method
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='evaluate the data')
    parser.add_argument('--map', required=True, help='landmarks')
    parser.add_argument('--execution', required=True, help='gts')
    parser.add_argument('--estimates', required=True, help='particle filter estimates')
    args = parser.parse_args()

    # Generate plots based on the loaded ground truth and estimates
    generate_plots(load_polygons(args.execution), load_polygons(args.estimates))


# calls main method to start
if __name__ == '__main__':
    main()
