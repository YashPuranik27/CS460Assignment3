import sys
import numpy as np
import matplotlib.pyplot as plt


def save_polygons(polygons, filename):
    np.save(filename, arr=polygons, allow_pickle=True)


def generate_landmarks(num):
    samples = []
    while len(samples) < num:
        new_sample = np.random.rand(2) * 2
        if all(np.all(new_sample != existing_sample) for existing_sample in samples):
            samples.append(new_sample)

    return np.array(samples)


def plot_landmarks(landmarks):
    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Generated Landmarks')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('generate landmarks')
        sys.exit(1)

    num_landmarks = int(sys.argv[1])
    filename = sys.argv[2]
    landmarks = generate_landmarks(num_landmarks)
    save_polygons(landmarks, filename)
    plot_landmarks(landmarks)
