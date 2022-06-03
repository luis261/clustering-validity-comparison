import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import random
from sklearn.preprocessing import MinMaxScaler


colors = ["#fc0303", "#8803fc", "#fcdb03", "#03fc30", "#458f61"]
plotting_marker_size = 1


class ColorException(Exception):
    pass

class XTooBigException(Exception):
    pass


def generate_benchmarking_suite(spherical=False):
    data, ideal_labels = generate_dataset(spherical=spherical)
    eps_choices_overfit = [x/800 for x in range(1, 6)]
    eps_choices_fit = [x/100 for x in range(1, 6)]
    eps_choices_underfit = [x/100 for x in range(11, 16)]
    eps_choices = eps_choices_overfit + eps_choices_fit + eps_choices_underfit
    labels_per_eps_choice = get_benchmarking_labels(data, eps_choices)
    return (data, ideal_labels, labels_per_eps_choice)

def get_benchmarking_labels(data, eps_choices, n_jobs=2):
    return [DBSCAN(eps=x, n_jobs=n_jobs).fit(data).labels_ for x in eps_choices]

def generate_dataset(spherical=False):
    noise_per_cluster_count = 10
    clusters_amount = random.randint(2, 5)
    noise_count = noise_per_cluster_count * clusters_amount
    data, labels = generate_clusters(clusters_amount, spherical=spherical)

    bounds = ((-25, 25), (-25, 25))
    data = np.concatenate((data, generate_noise(noise_count, bounds, data, 2.0)))
    noise_labels = np.array([-1 for _ in range(noise_count)])
    labels = np.concatenate((labels, noise_labels))
    data = MinMaxScaler().fit_transform(data)

    return (data, labels)

def plot_dataset(data, labels):
    if len(np.unique([x for x in labels if x != -1])) > len(colors):
        raise ColorException("The amount of clusters exceeds the amount of available colors")

    plt.figure()

    for i in range(data.shape[0]):
        if labels[i] == -1:
            plt.scatter(data[i, 0], data[i, 1], c="#000000", s=plotting_marker_size)
        else:
            plt.scatter(data[i, 0], data[i, 1], c=colors[labels[i]], s=plotting_marker_size)

    plt.show()

def generate_noise(count, bounds, data, distance):
    x_bounds = bounds[0]
    y_bounds = bounds[1]
    noise = [[random.uniform(x_bounds[0], x_bounds[1]), random.uniform(y_bounds[0], y_bounds[1])] for _ in range(count)]
    noise = [x for x in noise if point_far_enough_from_points(x, data, distance)]
    missing_noise = count - len(noise)
    while missing_noise > 0:
        noise += [[random.uniform(x_bounds[0], x_bounds[1]), random.uniform(y_bounds[0], y_bounds[1])] for _ in range(missing_noise)]
        noise = [x for x in noise if point_far_enough_from_points(x, data, distance)]
        missing_noise = count - len(noise)

    return np.array([np.array(x) for x in noise])

def generate_clusters(clusters_amount, spherical=False):
    clusters = []

    while len(clusters) < clusters_amount:

        overlap = True
        data, _ = make_blobs(n_samples=400,
            centers=1, center_box=(0, 0))

        if not spherical:
            matrix = generate_transformation_matrix()
            for i in range(0, data.shape[0]):
                data[i] = np.matmul(matrix, data[i])

        x_offset = random.uniform(-5.0, 5.0)
        for i in range(0, data.shape[0]):
            data[i][0] += x_offset

        y_offset = 0.0
        while overlap:
            if y_offset > 0.0:
                for i in range(0, data.shape[0]):
                    data[i][1] += y_offset

            overlap = False
            for cluster in clusters:
                if clusters_are_too_close(cluster, data, 2.0):
                    overlap = True
                    break

            y_offset = 0.8

        clusters.append(data)

    data = clusters[0]
    labels = np.array([0 for _ in range(clusters[0].shape[0])])
    for i in range(1, len(clusters)):
        data = np.concatenate((data, clusters[i]))
        labels = np.concatenate((labels, [i for _ in range(clusters[i].shape[0])]))

    return (data, labels)

def clusters_are_too_close(points0, points1, distance):
    for i in range(points0.shape[0]):
        if not point_far_enough_from_points(points0[i], points1, distance):
            return True

    return False

# uses euclidean distance
def point_far_enough_from_points(point, points, distance):
    for i in range(points.shape[0]):
        if np.linalg.norm(points[i] - point) < distance:
            return False

    return True

def generate_transformation_matrix():
    shape_modifier = 5.0
    return [[shape_modifier, 0],
            [0, 1/shape_modifier]]

def main():
    data, labels = generate_dataset(spherical=True)
    plot_dataset(data, labels)

    """
    data, ideal_labels, labelsets = generate_benchmarking_suite()
    plot_dataset(data, ideal_labels)
    for labelset in labelsets:
        try:
            plot_dataset(data, labelset)
        except ColorException as e:
            print(e)
    """


if __name__ == "__main__":
    main()