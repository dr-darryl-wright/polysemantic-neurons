import os
import sys
import argparse
import matplotlib

import numpy as np
import tensorflow as tf

sys.path.insert(0, '../utils')
import utils

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

def generate_mlp_neuron_vis_plots(model, save_path=None):

    # layer 1 analysis
    utils.generate_visualise_hidden_neurons_plot(model, 1, 256, (28, 28),
                                                 save_path=os.path.join(save_path, 'e1_vis.png'))

    # layer 2 analysis
    utils.generate_visualise_hidden_neurons_plot(model, 2, 256, (16, 16),
                                                 save_path=os.path.join(save_path, 'e2_vis.png'))

    # layer 3 analysis
    utils.generate_visualise_hidden_neurons_plot(model, 3, 256, (16, 16),
                                                 save_path=os.path.join(save_path, 'e3_vis.png'))

    # layer 4 analysis
    utils.generate_visualise_hidden_neurons_plot(model, 4, 256, (16, 16),
                                                 save_path=os.path.join(save_path, 'e4_vis.png'))


def generate_mlp_activations_plots(trial, x, y, model, i, e1, save_path=None):

    ps_term_metrics = []

    # layer 1 analysis
    embed1 = tf.keras.Model(inputs=i, outputs=e1)
    embed1.layers[1].set_weights(model.layers[1].get_weights())

    embed1_activations_by_class, embed1_classwise_ps_term_metric = \
        utils.generate_activations_plot(embed1, np.array(x), np.argmax(y, axis=1),
                                        save_path=os.path.join(save_path, 'e1_activations.png'))

    print('[{}] embed1 ps_term_metric : {}'.format(trial, embed1_classwise_ps_term_metric))
    ps_term_metrics.append(embed1_classwise_ps_term_metric)

    # layer 2 analysis
    embed2_input = tf.keras.layers.Input(shape=(28 * 28,))
    embed2_hidden1 = tf.keras.layers.Dense(256, activation='relu', name='embed2_hidden1')(embed2_input)
    embed2_output = tf.keras.layers.Dense(256, activation='relu', name='embed2_output')(embed2_hidden1)
    embed2 = tf.keras.Model(inputs=embed2_input, outputs=embed2_output)

    embed2.layers[1].set_weights(model.layers[1].get_weights())
    embed2.layers[2].set_weights(model.layers[2].get_weights())

    embed2_activations_by_class, embed2_classwise_ps_term_metric = \
        utils.generate_activations_plot(embed2, np.array(x), np.argmax(y, axis=1),
                                        save_path=os.path.join(save_path, 'e2_activations.png'))

    print('[{}] embed2 ps_term_metric : {}'.format(trial, embed2_classwise_ps_term_metric))
    ps_term_metrics.append(embed2_classwise_ps_term_metric)

    # layer 3 analysis
    embed3_input = tf.keras.layers.Input(shape=(28 * 28,))
    embed3_hidden1 = tf.keras.layers.Dense(256, activation='relu', name='embed3_hidden1')(embed3_input)
    embed3_hidden2 = tf.keras.layers.Dense(256, activation='relu', name='embed3_hidden2')(embed3_hidden1)
    embed3_output = tf.keras.layers.Dense(256, activation='relu', name='embed3_output')(embed3_hidden2)
    embed3 = tf.keras.Model(inputs=embed3_input, outputs=embed3_output)

    embed3.layers[1].set_weights(model.layers[1].get_weights())
    embed3.layers[2].set_weights(model.layers[2].get_weights())
    embed3.layers[3].set_weights(model.layers[3].get_weights())

    embed3_activations_by_class, embed3_classwise_ps_term_metric = \
        utils.generate_activations_plot(embed3, np.array(x), np.argmax(y, axis=1),
                                        save_path=os.path.join(save_path, 'e3_activations.png'))

    print('[{}] embed3 ps_term_metric : {}'.format(trial, embed3_classwise_ps_term_metric))
    ps_term_metrics.append(embed3_classwise_ps_term_metric)

    # layer 4 analysis
    embed4_input = tf.keras.layers.Input(shape=(28 * 28,))
    embed4_hidden1 = tf.keras.layers.Dense(256, activation='relu', name='embed4_hidden1')(embed4_input)
    embed4_hidden2 = tf.keras.layers.Dense(256, activation='relu', name='embed4_hidden2')(embed4_hidden1)
    embed4_hidden3 = tf.keras.layers.Dense(256, activation='relu', name='embed4_hidden3')(embed4_hidden2)
    embed4_output = tf.keras.layers.Dense(256, activation='relu', name='embed4_output')(embed4_hidden3)
    embed4 = tf.keras.Model(inputs=embed4_input, outputs=embed4_output)

    embed4.layers[1].set_weights(model.layers[1].get_weights())
    embed4.layers[2].set_weights(model.layers[2].get_weights())
    embed4.layers[3].set_weights(model.layers[3].get_weights())
    embed4.layers[4].set_weights(model.layers[4].get_weights())

    embed4_activations_by_class, embed4_classwise_ps_term_metric = \
        utils.generate_activations_plot(embed4, np.array(x), np.argmax(y, axis=1),
                                        save_path=os.path.join(save_path, 'e4_activations.png'))

    print('[{}] embed4 ps_term_metric : {}'.format(trial, embed4_classwise_ps_term_metric))
    ps_term_metrics.append(embed4_classwise_ps_term_metric)

    return ps_term_metrics


def run_mlp_layer_analysis(experiment, x_train, y_train, x_test, y_test):

    alphas = [0.0, 1.0, -1.0]

    for alpha in alphas:
        for trial in range(1,6):

            log_dir = '../../experiments/{}/alpha_{}/trial_{}/'.format(experiment, alpha, trial)

            try:
                os.makedirs(os.path.join(log_dir, 'layer_analysis', 'activations', 'train'))
                os.makedirs(os.path.join(log_dir, 'layer_analysis', 'activations', 'test'))
            except FileExistsError:
                pass

            _, (i, i2, e1, e2, e3, e4, o) = utils.get_mnist_clusterability_model()

            model = tf.keras.Model(inputs=i, outputs=o)
            model.load_weights(os.path.join(log_dir, 'inference_model.h5'))

            generate_mlp_neuron_vis_plots(model, save_path=os.path.join(log_dir, 'layer_analysis'))

            train_ps_term_metrics = generate_mlp_activations_plots(trial, x_train, y_train, model, i, e1,
                                                                   save_path=os.path.join(log_dir, 'layer_analysis',
                                                                                          'activations', 'train'))

            print('[{}] training set mean (std) ps_term_metrics : {}'.format(trial, np.mean(train_ps_term_metrics),
                                                                             np.std(train_ps_term_metrics)))

            test_ps_term_metrics = generate_mlp_activations_plots(trial, x_test, y_test, model, i, e1,
                                                                  save_path=os.path.join(log_dir, 'layer_analysis',
                                                                                         'activations', 'test'))

            print('[{}] test set mean (std) ps_term_metrics : {}'.format(trial, np.mean(test_ps_term_metrics),
                                                                         np.std(test_ps_term_metrics)))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment",
                        help="experiment to analyse. One of {mnist, fashion_mnist, mnist_with_pruning}",
                        type=str, required=True)

    args = parser.parse_args()
    experiment = args.experiment

    if experiment == 'mnist':
        (x_train, y_train), (x_test, y_test) = utils.get_flattened_mnist()
        run_mlp_layer_analysis(experiment, x_train, y_train, x_test, y_test)
    elif experiment == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = utils.get_flattened_fashion_mnist()
        run_mlp_layer_analysis(experiment, x_train, y_train, x_test, y_test)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
