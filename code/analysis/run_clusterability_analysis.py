import os
import sys
import csv
import tqdm
import argparse
import matplotlib

import numpy as np
import tensorflow as tf
import scipy.stats as ss
import matplotlib.pyplot as plt

sys.path.insert(0, '../utils')
import utils

sys.path.insert(0, '../../../clusterability_in_neural_networks/')
from src import spectral_cluster_model, train_nn
from src.utils import compute_pvalue

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)


def read_results(filename):

    n_cuts = []
    z_scores = []
    left_p_values = []
    with open(filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            n_cuts.append(float(row['n_cut']))
            z_scores.append(float(row['z_score']))
            left_p_values.append(float(row['left_p_value']))
    return n_cuts, z_scores, left_p_values

# def run_clusterability_analysis(experiment, x_train, y_train, x_test, y_test):
#
#     alphas = [0.0, 1.0, -1.0]
#
#     for alpha in alphas:
#         log_dir = '../../experiments/{}/alpha_{}/'.format(experiment, alpha)
#         with open(os.path.join(log_dir, 'clusterability_with_pruning_results.csv'), 'w') as outfile:
#             writer = csv.writer(outfile)
#         for trial in range(1, 6):
#
#             _, (i, i2, e1, e2, e3, e4, o) = utils.get_mnist_clusterability_model()
#
#             model = tf.keras.Model(inputs=i, outputs=o)
#             model.load_weights(os.path.join(log_dir, 'trial_{}'.format(trial), 'inference_model.h5'))
#
#                     weights = [w for w in model.get_weights() if len(w.shape) == 2]  # skip biases
#                     adj_mat = spectral_cluster_model.weights_to_graph(weights)
#                     clustering_labels = spectral_cluster_model.cluster_net(k, adj_mat, 'arpack',
#                                                                            assign_labels='kmeans')
#                     n_cut = spectral_cluster_model.compute_ncut(adj_mat, clustering_labels, epsilon=1e-8, verbose=True)
#                     n_cuts.append(n_cut)
#
#                     shuffled_n_cuts = []
#                     for j in tqdm.tqdm(range(50)):
#                         shuffled_weights = [spectral_cluster_model.shuffle_weights(w) for w in weights[:]]
#                         shuffled_adj_mat = spectral_cluster_model.weights_to_graph(shuffled_weights)
#                         shuffled_clustering_labels = spectral_cluster_model.cluster_net(k, shuffled_adj_mat, 'arpack',
#                                                                                         assign_labels='kmeans')
#                         shuffled_n_cut = spectral_cluster_model.compute_ncut(shuffled_adj_mat,
#                                                                              shuffled_clustering_labels,
#                                                                              epsilon=1e-8, verbose=False)
#                         shuffled_n_cuts.append(shuffled_n_cut)
#
#                     z_score = (n_cut - np.mean(shuffled_n_cuts)) / np.std(shuffled_n_cuts)
#                     z_scores.append(z_score)
#
#                     left_p_value = compute_pvalue(n_cut, shuffled_n_cuts)
#                     left_p_values.append(left_p_value)
#
#                     writer.writerow([str(trial), str(n_cut), str(z_score), str(left_p_value)])
#
#         plt.scatter(n_cuts, z_scores, marker='o', s=500,
#                     color=utils.colour_scheme[alphas.index(alpha)],
#                     label='alpha={}'.format(alpha))
#
#         for i in range(len(left_p_values)):
#             plt.text(n_cuts[i]+0.01, z_scores[i]+0.01, '{0:.3f}'.format(left_p_values[i]))
#
#     plt.grid()
#     #plt.xlim(8.2, 11.0)
#     #plt.ylim(-10, 15)
#     plt.xlabel('N-cut')
#     plt.ylabel('Z-score')
#     plt.title('MLP Clusterability, Penalise Polysemantic Neurons')
#     plt.legend(loc='upper left')
#     plt.savefig('../../experiments/mnist/clusterability_k{}.png'.format(k), bbox_inches='tight')


def run_mnist(overwrite=False):

    alphas = [0.0, 1.0, -1.0]
    k = 12

    _, (i, _, _, _, _, _, o) = utils.get_mnist_clusterability_model()

    model = tf.keras.Model(inputs=i, outputs=o)

    fig = plt.figure(figsize=(10,10))
    for alpha in alphas:
        n_cuts = []
        z_scores = []
        left_p_values = []
        results_path = os.path.join('../../experiments/mnist/alpha_{}/'.format(alpha),
                                    'clusterability_results_k{}.csv'.format(k))
        try:
            if overwrite:
                raise OSError
            n_cuts, z_scores, left_p_values = read_results(results_path)
        except OSError:
            with open(results_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['trial', 'n_cut', 'z_score', 'left_p_value'])
                for trial in range(1, 6):
                    log_dir = '../../experiments/mnist/alpha_{}/trial_{}/'.format(alpha, trial)
                    model.load_weights(os.path.join(log_dir, 'inference_model.h5'))

                    weights = [w for w in model.get_weights() if len(w.shape) == 2]  # skip biases
                    adj_mat = spectral_cluster_model.weights_to_graph(weights)
                    clustering_labels = spectral_cluster_model.cluster_net(k, adj_mat, 'arpack',
                                                                           assign_labels='kmeans')
                    n_cut = spectral_cluster_model.compute_ncut(adj_mat, clustering_labels, epsilon=1e-8, verbose=True)
                    n_cuts.append(n_cut)

                    shuffled_n_cuts = []
                    for j in tqdm.tqdm(range(50)):
                        shuffled_weights = [spectral_cluster_model.shuffle_weights(w) for w in weights[:]]
                        shuffled_adj_mat = spectral_cluster_model.weights_to_graph(shuffled_weights)
                        shuffled_clustering_labels = spectral_cluster_model.cluster_net(k, shuffled_adj_mat, 'arpack',
                                                                                        assign_labels='kmeans')
                        shuffled_n_cut = spectral_cluster_model.compute_ncut(shuffled_adj_mat,
                                                                             shuffled_clustering_labels,
                                                                             epsilon=1e-8, verbose=False)
                        shuffled_n_cuts.append(shuffled_n_cut)

                    z_score = (n_cut - np.mean(shuffled_n_cuts)) / np.std(shuffled_n_cuts)
                    z_scores.append(z_score)

                    left_p_value = compute_pvalue(n_cut, shuffled_n_cuts)
                    left_p_values.append(left_p_value)

                    writer.writerow([str(trial), str(n_cut), str(z_score), str(left_p_value)])

        plt.scatter(n_cuts, z_scores, marker='o', s=250,
                    color=utils.colour_scheme[alphas.index(alpha)],
                    label='alpha={}'.format(alpha))

        for i in range(len(left_p_values)):
            plt.text(n_cuts[i]+0.01, z_scores[i]+0.01, '{0:.3f}'.format(left_p_values[i]))

    plt.grid()
    #plt.xlim(8.2, 11.0)
    #plt.ylim(-10, 15)
    plt.xlabel('N-cut')
    plt.ylabel('Z-score')
    plt.title('MLP Clusterability, Penalise Polysemantic Neurons')
    plt.legend(loc='upper left')
    plt.savefig('../../experiments/mnist/clusterability_k{}.png'.format(k), bbox_inches='tight')

    return n_cuts, z_scores, left_p_values


def run_mnist_with_pruning(overwrite):

    alphas = [0.0, 1.0, -1.0]

    _, (i, _, _, _, _, _, o) = utils.get_mnist_clusterability_model()

    model = tf.keras.Model(inputs=i, outputs=o)

    fig = plt.figure(figsize=(10,10))
    for alpha in alphas:
        n_cuts = []
        z_scores = []
        left_p_values = []
        results_path = os.path.join('../../experiments/mnist/alpha_{}/'.format(alpha),
                                    'clusterability_with_pruning_results.csv')
        try:
            if overwrite:
                raise OSError
            n_cuts, z_scores, left_p_values = read_results(results_path)
        except OSError:
            with open(results_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['trial', 'n_cut', 'z_score', 'left_p_value'])
                for trial in range(1, 6):
                    log_dir = '../../experiments/mnist/alpha_{}/trial_{}/pruned/'.format(alpha, trial)
                    model.load_weights(os.path.join(log_dir, 'inference_model.h5'))

                    weights = [w for w in model.get_weights() if len(w.shape) == 2]  # skip biases
                    adj_mat = spectral_cluster_model.weights_to_graph(weights)
                    clustering_labels = spectral_cluster_model.cluster_net(12, adj_mat, 'arpack',
                                                                           assign_labels='kmeans')
                    n_cut = spectral_cluster_model.compute_ncut(adj_mat, clustering_labels, epsilon=1e-8, verbose=True)
                    n_cuts.append(n_cut)

                    shuffled_n_cuts = []
                    for j in tqdm.tqdm(range(50)):
                        shuffled_weights = [spectral_cluster_model.shuffle_weights(w) for w in weights[:]]
                        shuffled_adj_mat = spectral_cluster_model.weights_to_graph(shuffled_weights)
                        shuffled_clustering_labels = spectral_cluster_model.cluster_net(12, shuffled_adj_mat, 'arpack',
                                                                                        assign_labels='kmeans')
                        shuffled_n_cut = spectral_cluster_model.compute_ncut(shuffled_adj_mat,
                                                                             shuffled_clustering_labels,
                                                                             epsilon=1e-8, verbose=False)
                        shuffled_n_cuts.append(shuffled_n_cut)

                    z_score = (n_cut - np.mean(shuffled_n_cuts)) / np.std(shuffled_n_cuts)
                    z_scores.append(z_score)

                    left_p_value = compute_pvalue(n_cut, shuffled_n_cuts)
                    left_p_values.append(left_p_value)

                    writer.writerow([str(trial), str(n_cut), str(z_score), str(left_p_value)])

        plt.scatter(n_cuts, z_scores, marker='o', s=500,
                    color=utils.colour_scheme[alphas.index(alpha)],
                    label='alpha={}, pruning'.format(alpha))

        for i in range(len(left_p_values)):
            plt.text(n_cuts[i]+0.01, z_scores[i]+0.01, '{0:.3f}'.format(left_p_values[i]))

    plt.grid()
    plt.xlim(9.0, 11.0)
    plt.ylim(-20, 15)
    plt.xlabel('N-cut')
    plt.ylabel('Z-score')
    plt.title('MLP Clusterability, Penalise Polysemantic Neurons with Pruning')
    plt.legend()
    plt.savefig('../../experiments/mnist/clusterability_with_pruning.png', bbox_inches='tight')

    return n_cuts, z_scores, left_p_values


### Fashion MNIST ###

def run_fashion_mnist(overwrite=False):

    alphas = [0.0, 1.0, -1.0]
    k = 12

    _, (i, _, _, _, _, _, o) = utils.get_mnist_clusterability_model()

    model = tf.keras.Model(inputs=i, outputs=o)

    fig = plt.figure(figsize=(10,10))
    for alpha in alphas:
        n_cuts = []
        z_scores = []
        left_p_values = []
        results_path = os.path.join('../../experiments/fashion_mnist/alpha_{}/'.format(alpha),
                                    'clusterability_results_k{}.csv'.format(k))
        try:
            if overwrite:
                raise OSError
            n_cuts, z_scores, left_p_values = read_results(results_path)
        except OSError:
            with open(results_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['trial', 'n_cut', 'z_score', 'left_p_value'])
                for trial in range(1, 6):
                    log_dir = '../../experiments/fashion_mnist/alpha_{}/trial_{}/'.format(alpha, trial)
                    model.load_weights(os.path.join(log_dir, 'inference_model.h5'))

                    weights = [w for w in model.get_weights() if len(w.shape) == 2]  # skip biases
                    adj_mat = spectral_cluster_model.weights_to_graph(weights)
                    clustering_labels = spectral_cluster_model.cluster_net(k, adj_mat, 'arpack',
                                                                           assign_labels='kmeans')
                    n_cut = spectral_cluster_model.compute_ncut(adj_mat, clustering_labels, epsilon=1e-8, verbose=True)
                    n_cuts.append(n_cut)

                    shuffled_n_cuts = []
                    for j in tqdm.tqdm(range(50)):
                        shuffled_weights = [spectral_cluster_model.shuffle_weights(w) for w in weights[:]]
                        shuffled_adj_mat = spectral_cluster_model.weights_to_graph(shuffled_weights)
                        shuffled_clustering_labels = spectral_cluster_model.cluster_net(k, shuffled_adj_mat, 'arpack',
                                                                                        assign_labels='kmeans')
                        shuffled_n_cut = spectral_cluster_model.compute_ncut(shuffled_adj_mat,
                                                                             shuffled_clustering_labels,
                                                                             epsilon=1e-8, verbose=False)
                        shuffled_n_cuts.append(shuffled_n_cut)

                    z_score = (n_cut - np.mean(shuffled_n_cuts)) / np.std(shuffled_n_cuts)
                    z_scores.append(z_score)

                    left_p_value = compute_pvalue(n_cut, shuffled_n_cuts)
                    left_p_values.append(left_p_value)

                    writer.writerow([str(trial), str(n_cut), str(z_score), str(left_p_value)])

        plt.scatter(n_cuts, z_scores, marker='o', s=500,
                    color=utils.colour_scheme[alphas.index(alpha)],
                    label='alpha={}'.format(alpha))

        for i in range(len(left_p_values)):
            plt.text(n_cuts[i]+0.01, z_scores[i]+0.01, '{0:.3f}'.format(left_p_values[i]))

    plt.grid()
    #plt.xlim(8.2, 11.0)
    #plt.ylim(-10, 15)
    plt.xlabel('N-cut')
    plt.ylabel('Z-score')
    plt.title('MLP Clusterability, Penalise Polysemantic Neurons')
    plt.legend(loc='upper left')
    plt.savefig('../../experiments/fashion_mnist/clusterability_k{}.png'.format(k), bbox_inches='tight')

    return n_cuts, z_scores, left_p_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment",
                        help="experiment to analyse. One of {mnist, fashion_mnist, mnist_with_pruning}",
                        type=str, required=True)
    parser.add_argument("-ow", "--overwrite",
                        help="whether to overwrite previous results",
                        action='store_true')

    args = parser.parse_args()
    experiment = args.experiment
    overwrite = args.overwrite

    if experiment == 'mnist':
        run_mnist(overwrite)
    elif experiment == 'fashion_mnist':
        run_fashion_mnist(overwrite)
    elif experiment == 'mnist_with_pruning':
        run_mnist_with_pruning(overwrite)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
