import sys
import numpy as np
import matplotlib.pyplot as plt


def save_polygons(polygons, filename):
    np.save(filename, arr=polygons, allow_pickle=True)


def generate_landmarks(num):
    samples = []
    while len(samples) < num:
        # Generate samples within the range [0, 2) for both x and y coordinates
        new_sample = np.random.rand(2) * 2
        if all(np.all(new_sample != existing_sample) for existing_sample in samples):
            samples.append(new_sample)

    return np.array(samples)


def plot_landmarks(landmarks):
    plt.scatter(landmarks[:, 0], landmarks[:, 1])  # Plot the landmarks
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Generated Landmarks')
    plt.xlim(0, 2)  # Set x-axis limit
    plt.ylim(0, 2)  # Set y-axis limit
    plt.grid(True)  # grid for better visibility
    plt.gca().set_aspect('equal', adjustable='box')  # equal aspect ratio
    plt.show()


# to run: python landmark_generation.py [number of landmarks as an integer] [name of file with .npy]
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('to run: python landmark_generation.py [number of landmarks as an integer] [name of file with .npy]')
        sys.exit(1)

    num_landmarks = int(sys.argv[1])
    filename = sys.argv[2]
    landmarks = generate_landmarks(num_landmarks)
    save_polygons(landmarks, filename)
    plot_landmarks(landmarks)  # Call the plot function
