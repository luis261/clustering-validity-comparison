from custom_metrics import calc_jaccard_coeff
from hdbscan.validity import validity_index
from generate_benchmarking_suite import generate_benchmarking_suite, plot_dataset, ColorException
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def mean(chain):
    return sum(chain)/len(chain)

def calc_noise_adjusted_score(metric, data, labels, inverse=False):
    indices_without_noise = [i for i in range(len(labels)) if labels[i] != -1]
    data_without_noise = np.array([data[i] for i in indices_without_noise])
    labels_without_noise = [labels[i] for i in indices_without_noise]
    multiplier = len(data)/len(data_without_noise) if inverse else len(data_without_noise)/len(data)

    return metric(data_without_noise, labels_without_noise) * multiplier

def generate_noise_handling_wrapper(metric, inverse=False):
    return lambda data, labels : calc_noise_adjusted_score(metric, data, labels, inverse=inverse)

def benchmark_metrics(data, ideal_labels, labelsets, metrics, all_details=False, max_plots=30):
    plot_count = 0
    notified = False
    similarities_to_ideal_labels = []
    ratings_per_metric = [[] for _ in metrics]
    for labelset in labelsets:
        # if there are less than 2 clusters, skip the given labels
        if len(np.unique([x for x in labelset if x != -1])) < 2:
            continue

        result_buffer = []
        for i in range(len(metrics)):
            # the validation with DBCV raises ValueErrors for clusters whose points have 0 distance between them
            # (edge case that does not occur often)
            # see here: https://github.com/scikit-learn-contrib/hdbscan/issues/127#issuecomment-689391326
            try:
                result_buffer.append(metrics[i](data, labelset))
            except ValueError as e:
                print("A ValueError occurred during the validation calculation:")
                print(str(e))
                result_buffer = []
                break

        if len(result_buffer) == 0:
            print("The result buffer is empty, no new values will be appended")
            continue

        for i in range(len(result_buffer)):
            ratings_per_metric[i].append(result_buffer[i])

        similarities_to_ideal_labels.append(calc_jaccard_coeff(ideal_labels, labelset))

        if all_details:
            if not notified:
                print("These are the scores of the metrics for the clustering which will be plotted next:")
            else:
                print("These are the scores of the metrics:")

            print(result_buffer)
            print("Similarity score:")
            print(similarities_to_ideal_labels[-1])

            if plot_count < max_plots:
                try:
                    plot_dataset(data, labelset)
                    plot_count += 1
                except ColorException as e:
                    print(str(e) + ": plotting had to be skipped")
            elif not notified:
                print("Skipping the remaining plots")
                notified = True

    return [pearsonr(ratings_per_metric[i], similarities_to_ideal_labels)[0] for i in range(len(metrics))]

def summarize_benchmarks(count, spherical=False, visualize=False, all_details=False, max_plots_per_dataset=30):
    metrics = [validity_index]
    non_mrd_dbcv = lambda data, labels : validity_index(data, labels, mst_euclid_only=True)
    metrics.append(non_mrd_dbcv)
    non_noise_handling_metrics = [silhouette_score, calinski_harabasz_score]
    inverse_non_noise_handling_metrics = [davies_bouldin_score]
    metrics += [generate_noise_handling_wrapper(metric) for metric in non_noise_handling_metrics]
    metrics += [generate_noise_handling_wrapper(metric, inverse=True) for metric in inverse_non_noise_handling_metrics]

    benchmark_results_per_metric = [[] for _ in metrics]
    for _ in range(count):
        print("Generating the next benchmarking suite which will be used to compare the metrics")
        data, ideal_labels, labelsets = generate_benchmarking_suite(spherical=spherical)
        if visualize:
            plot_dataset(data, ideal_labels)

        print("Using the generated benchmarking suite to compare the metrics")
        results = benchmark_metrics(data, ideal_labels, labelsets, metrics, all_details=all_details, max_plots=max_plots_per_dataset)
        for i in range(len(results)):
            benchmark_results_per_metric[i].append(results[i])

    print("The histogramms of the correlations per metric will be plotted next")
    for result in benchmark_results_per_metric:
        plt.hist(result, bins=50)
        plt.show()

    return [mean(x) for x in benchmark_results_per_metric]

def main():
    dataset_count = 50
    print("Benchmarking the given metrics regarding the application on spherical data")
    result_spherical = summarize_benchmarks(dataset_count, spherical=True)
    print("Benchmarking the given metrics regarding the application on non-spherical data")
    result_non_spherical = summarize_benchmarks(dataset_count)
    print("Overall correlations per metric (spherical & non-spherical)")
    print([(result_spherical[i] + result_non_spherical[i])/2 for i in range(len(result_spherical))])


if __name__ == "__main__":
    main()