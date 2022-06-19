import os
import sys
import tqdm
import argparse
import warnings
import itertools
import matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

sys.path.insert(0, '../utils')
from utils import get_flattened_mnist, get_mnist_clusterability_model#, get_model

sys.path.insert(0, '../../../clusterability_in_neural_networks/')
from src.visualization import extract_layer_widths
from src.lesion.experimentation import (_layers_labels_gen, _damaged_neurons_gen, _damage_neurons, _evaluate,
                                        _extract_layer_label_metadata, _flatten_single_damage)
from src import spectral_cluster_model, train_nn
from src.utils import extract_weights

from src.lesion import (do_lesion_hypo_tests,
                        plot_all_damaged_clusters,
                        plot_overall_damaged_clusters)

def run_lesion_experiment(model, x, y, weights, biases, layer_widths,
                          labels, ignore_layers=False, to_shuffle=False):

    layers_labels_iterable = _layers_labels_gen('mlp', layer_widths, labels,
                                                ignore_layers, to_shuffle)

    damage_results = []

    for neurons_in_layers in _damaged_neurons_gen('mlp', layer_widths, labels, ignore_layers,
                                                  to_shuffle, n_way=1, n_way_type='joint', verbose=False):
        _damage_neurons(neurons_in_layers, model, weights, biases, 'mlp', inplace=True)

        result = _evaluate(model, np.array(x), np.array(np.argmax(y, axis=1)), 'classification')  # , masks)
        result['labels_in_layers'] = tuple((layer_id, label) for layer_id, label, _, _ in neurons_in_layers)
        damage_results.append(result)

    return damage_results


def run_local_specialization_analysis(experiment, x, y, n_clusters=12, n_shuffles=50, network_type='mlp'):

    #_, (i, _, _, _, _, _, o) = get_mnist_clusterability_model()

    #model = tf.keras.Model(inputs=i, outputs=o)

    n_hidden_neurons = 256
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28 * 28,), name='input'))
    model.add(tf.keras.layers.Dense(n_hidden_neurons, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dense(n_hidden_neurons, activation='relu', name='dense_2'))
    model.add(tf.keras.layers.Dense(n_hidden_neurons, activation='relu', name='dense_3'))
    model.add(tf.keras.layers.Dense(n_hidden_neurons, activation='relu', name='dense_4'))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name='softmax'))

    model.load_weights('../../notebooks/mnist_penalise_polysemantic_neurons_all_classes_clusterability_weights_epochs_100.h5')
    #model = get_model(experiment, trained=True)

    weights, biases = extract_weights(model, with_bias=True)

    layer_widths = extract_layer_widths(weights)

    adj_mat = spectral_cluster_model.weights_to_graph(weights)

    labels = spectral_cluster_model.cluster_net(n_clusters, adj_mat, 'arpack',
                                                assign_labels='kmeans')

    evaluation = _evaluate(model, np.array(x), np.array(np.argmax(y, axis=1)), 'classification')

    metadata = _extract_layer_label_metadata(network_type, layer_widths, labels, ignore_layers=False)

    true_results = run_lesion_experiment(model, x, y, weights, biases, layer_widths,
                                         labels, ignore_layers=False, to_shuffle=False)

    all_random_results = []
    for i in tqdm.tqdm(range(n_shuffles)):
        random_results = run_lesion_experiment(model, x, y, weights, biases, layer_widths,
                                               labels, ignore_layers=False, to_shuffle=True)

        all_random_results.append(random_results)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        axes = plot_all_damaged_clusters(true_results, all_random_results, metadata,
                                         evaluation, title=experiment)

        axes.savefig('./test1.png')

        axes = plot_overall_damaged_clusters(true_results, all_random_results, metadata,
                                             evaluation, title=experiment)

        axes.savefig('./test2.png')

        hypo_results = do_lesion_hypo_tests(evaluation, true_results, all_random_results)
        mean_percentiles = hypo_results['mean_percentiles']
        range_percentiles = hypo_results['range_percentiles']
        chi2_p_means = hypo_results['chi2_p_means']
        chi2_p_ranges = hypo_results['chi2_p_ranges']
        combined_p_means = hypo_results['combined_p_means']
        combined_p_ranges = hypo_results['combined_p_ranges']
        print(f'{experiment}')
        print(f'Mean percentiles: {mean_percentiles}')
        print(f'Range percentiles: {range_percentiles}')
        print(f'chi2 mean p: {chi2_p_means}')
        print(f'chi2 range p: {chi2_p_ranges}')
        print(f'combined mean p: {combined_p_means}')
        print(f'combined range p: {combined_p_ranges}')
        print()


def main():
    _, (x, y) = get_flattened_mnist()
    run_local_specialization_analysis('test', x, y, n_clusters=12, n_shuffles=1, network_type='mlp')

if __name__ == '__main__':
    main()