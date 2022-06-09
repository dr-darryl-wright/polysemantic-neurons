import os
import sys
import yaml
import argparse
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


sys.path.insert(0, '../utils')
import utils


### MNIST ###
def run_mnist(batch_size, epochs, alpha, lr):

    (x_train, y_train), (x_test, y_test) = utils.get_flattened_mnist()

    # number of training examples to use. Must be multiple of batch_size as the current loss cannot deal with batches
    # where the number of examples are not a multiple of batch_size. This should only result in negligible differences
    # to the clusterability paper
    limit = (batch_size * (len(x_train) // batch_size))
    test_limit = (batch_size * (len(x_test) // batch_size))  # Must also be multiple of batch_size

    c0, c1 = [], []
    for c in list(itertools.combinations(range(batch_size), 2)):
        c0.append(c[0])
        c1.append(c[1])

    test_set_accs = []
    for trial in range(1, 6):

        log_dir = '../../experiments/mnist/alpha_{}/trial_{}/'.format(alpha, trial)

        model, (i, i2, e1, e2, e3, e4, o) = utils.get_mnist_clusterability_model()

        model.add_loss(utils.get_mnist_clusterability_loss(i2, o, e1, e2, e3, e4,
                                                           np.array(c0, dtype='int32'),
                                                           np.array(c1, dtype='int32'),
                                                           alpha=alpha))

        # clusterability paper uses Adam with lr=0.001
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=None)

        model.add_metric(utils.cce_metric(i2, o), name='cce', aggregation='mean')
        model.add_metric(utils.get_mnist_clusterability_loss(i2, o, e1, e2, e3, e4, c0, c1, alpha),
                         name='p_loss', aggregation='mean')
        model.add_metric(utils.acc_metric(i2, o), name='acc', aggregation='mean')
        model.add_metric(utils.ps_term_metric(i2, e1, c0, c1), name='ps1', aggregation='mean')
        model.add_metric(utils.ps_term_metric(i2, e2, c0, c1), name='ps2', aggregation='mean')
        model.add_metric(utils.ps_term_metric(i2, e3, c0, c1), name='ps3', aggregation='mean')
        model.add_metric(utils.ps_term_metric(i2, e4, c0, c1), name='ps4', aggregation='mean')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=10,
                write_images=True,
                embeddings_freq=10
        )

        history = model.fit([np.array(x_train)[:limit], y_train[:limit]],
                            y=None,
                            validation_data=([np.array(x_test)[:test_limit],
                                              y_test[:test_limit]], None),
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[tensorboard_callback]
        )

        utils.generate_history_plots(history, ['loss', 'cce', 'p_loss', 'acc', 'ps1', 'ps2', 'ps3', 'ps4'],
                                     epochs, save_path=os.path.join(log_dir, 'history.png'))

        # measure test set accuracy
        inference = tf.keras.Model(inputs=i, outputs=o)
        inference.save_weights(os.path.join(log_dir, 'inference_model.h5'))
        y_pred_test = inference.predict(np.array(x_test))
        print('[{}] test set accuracy : {}'.format(trial, np.mean(utils.acc_metric(y_test, y_pred_test))))
        test_set_accs.append(np.mean(utils.acc_metric(y_test, y_pred_test)))

    print('test_set_accs: {}'.format(test_set_accs))
    print('mean test set accuracy across trials: {}'.format(np.mean(test_set_accs)))


def run_mnist_with_pruning(batch_size, epochs, alpha, lr):
    """
    For pruning, our initial sparsity is 0.5, our final sparsity is 0.9, the pruning frequency is 10 steps, and we use
    a cubic pruning schedule (see Zhu and Gupta (2017)). Initial and final sparsities were chosen due to their use in
    the TensorFlow Model Optimization Tutorial.
    """
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    (x_train, y_train), (x_test, y_test) = utils.get_flattened_mnist()

    # number of training examples to use. Must be multiple of batch_size as the current loss cannot deal with batches
    # where the number of examples are not a multiple of batch_size. This should only result in negligible differences
    # to the clusterability paper
    limit = (batch_size * (len(x_train) // batch_size))
    test_limit = (batch_size * (len(x_test) // batch_size))  # Must also be multiple of batch_size

    end_step = np.ceil(limit // batch_size) * epochs

    c0, c1 = [], []
    for c in list(itertools.combinations(range(batch_size), 2)):
        c0.append(c[0])
        c1.append(c[1])

    test_set_accs = []
    for trial in range(1, 6):

        log_dir = '../../experiments/mnist/alpha_{}/trial_{}/pruned/'.format(alpha, trial)
        saved_model_dir = '../../experiments/mnist/alpha_{}/trial_{}/'.format(alpha, trial)

        #model, (i, i2, e1, e2, e3, e4, o) = utils.get_mnist_clusterability_model()
        model, _ = utils.get_mnist_clusterability_model()

        model.load_weights(os.path.join(saved_model_dir, 'inference_model.h5'))

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                     final_sparsity=0.90,
                                                                     begin_step=0,
                                                                     frequency=10,
                                                                     end_step=end_step)
        }

        model_for_pruning = prune_low_magnitude(model, **pruning_params)

        model_for_pruning.summary()

        i = model_for_pruning.get_layer('input_1')
        i2 = model_for_pruning.get_layer('input_2')
        e1 = model_for_pruning.get_layer('prune_low_magnitude_dense_1')(i)
        e2 = model_for_pruning.get_layer('prune_low_magnitude_dense_2')(e1)
        e3 = model_for_pruning.get_layer('prune_low_magnitude_dense_3')(e2)
        e4 = model_for_pruning.get_layer('prune_low_magnitude_dense_4')(e3)
        o = model_for_pruning.get_layer('prune_low_magnitude_softmax')(e4)

        model_for_pruning = tf.keras.Model(inputs=[i, i2], outputs=o)

        model_for_pruning.add_loss(utils.get_mnist_clusterability_loss(i2, o, e1, e2, e3, e4,
                                                                       np.array(c0, dtype='int32'),
                                                                       np.array(c1, dtype='int32'),
                                                                       alpha=alpha))
        # clusterability paper uses Adam with lr=0.001
        model_for_pruning.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=None)

        model_for_pruning.add_metric(utils.cce_metric(i2, o), name='cce', aggregation='mean')
        model_for_pruning.add_metric(utils.get_mnist_clusterability_loss(i2, o, e1, e2, e3, e4, c0, c1, alpha),
                         name='p_loss', aggregation='mean')
        model_for_pruning.add_metric(utils.acc_metric(i2, o), name='acc', aggregation='mean')
        model_for_pruning.add_metric(utils.ps_term_metric(i2, e1, c0, c1), name='ps1', aggregation='mean')
        model_for_pruning.add_metric(utils.ps_term_metric(i2, e2, c0, c1), name='ps2', aggregation='mean')
        model_for_pruning.add_metric(utils.ps_term_metric(i2, e3, c0, c1), name='ps3', aggregation='mean')
        model_for_pruning.add_metric(utils.ps_term_metric(i2, e4, c0, c1), name='ps4', aggregation='mean')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=10,
                write_images=True,
                embeddings_freq=10
        )
        history = model_for_pruning.fit([
            np.array(x_train)[:limit], y_train[:limit]],
            y=None,
            validation_data=([np.array(x_test)[:test_limit],
                              y_test[:test_limit]], None),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tensorboard_callback]
        )

        utils.generate_history_plots(history, ['loss', 'cce', 'p_loss', 'acc', 'ps1', 'ps2', 'ps3', 'ps4'],
                                     epochs, save_path=os.path.join(log_dir, 'history.png'))

        # measure test set accuracy
        inference = tf.keras.Model(inputs=i, outputs=o)
        inference.save_weights(os.path.join(log_dir, 'inference_model.h5'))
        y_pred_test = inference.predict(np.array(x_test))
        print('[{}] test set accuracy : {}'.format(trial, np.mean(utils.acc_metric(y_test, y_pred_test))))
        test_set_accs.append(np.mean(utils.acc_metric(y_test, y_pred_test)))

    print('test_set_accs: {}'.format(test_set_accs))
    print('mean test set accuracy across trials: {}'.format(np.mean(test_set_accs)))


### Fashion MNIST ###

def run_fashion_mnist(batch_size, epochs, alpha, lr):

    (x_train, y_train), (x_test, y_test) = utils.get_flattened_fashion_mnist()

    # number of training examples to use. Must be multiple of batch_size as the current loss cannot deal with batches
    # where the number of examples are not a multiple of batch_size. This should only result in negligible differences
    # to the clusterability paper
    limit = (batch_size * (len(x_train) // batch_size))
    test_limit = (batch_size * (len(x_test) // batch_size))  # Must also be multiple of batch_size

    c0, c1 = [], []
    for c in list(itertools.combinations(range(batch_size), 2)):
        c0.append(c[0])
        c1.append(c[1])

    test_set_accs = []
    for trial in range(1, 6):

        log_dir = '../../experiments/fashion_mnist/alpha_{}/trial_{}/'.format(alpha, trial)

        model, (i, i2, e1, e2, e3, e4, o) = utils.get_mnist_clusterability_model()

        model.add_loss(utils.get_mnist_clusterability_loss(i2, o, e1, e2, e3, e4,
                                                           np.array(c0, dtype='int32'),
                                                           np.array(c1, dtype='int32'),
                                                           alpha=alpha))

        # clusterability paper uses Adam with lr=0.001
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=None)

        model.add_metric(utils.cce_metric(i2, o), name='cce', aggregation='mean')
        model.add_metric(utils.get_mnist_clusterability_loss(i2, o, e1, e2, e3, e4, c0, c1, alpha),
                         name='p_loss', aggregation='mean')
        model.add_metric(utils.acc_metric(i2, o), name='acc', aggregation='mean')
        model.add_metric(utils.ps_term_metric(i2, e1, c0, c1), name='ps1', aggregation='mean')
        model.add_metric(utils.ps_term_metric(i2, e2, c0, c1), name='ps2', aggregation='mean')
        model.add_metric(utils.ps_term_metric(i2, e3, c0, c1), name='ps3', aggregation='mean')
        model.add_metric(utils.ps_term_metric(i2, e4, c0, c1), name='ps4', aggregation='mean')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=10,
                write_images=True,
                embeddings_freq=10
        )

        history = model.fit([np.array(x_train)[:limit], y_train[:limit]],
                            y=None,
                            validation_data=([np.array(x_test)[:test_limit],
                                              y_test[:test_limit]], None),
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[tensorboard_callback]
        )

        utils.generate_history_plots(history, ['loss', 'cce', 'p_loss', 'acc', 'ps1', 'ps2', 'ps3', 'ps4'],
                                     epochs, save_path=os.path.join(log_dir, 'history.png'))

        # measure test set accuracy
        inference = tf.keras.Model(inputs=i, outputs=o)
        inference.save_weights(os.path.join(log_dir, 'inference_model.h5'))
        y_pred_test = inference.predict(np.array(x_test))
        print('[{}] test set accuracy : {}'.format(trial, np.mean(utils.acc_metric(y_test, y_pred_test))))
        test_set_accs.append(np.mean(utils.acc_metric(y_test, y_pred_test)))

    print('test_set_accs: {}'.format(test_set_accs))
    print('mean test set accuracy across trials: {}'.format(np.mean(test_set_accs)))


def run_fashion_mnist_with_pruning(batch_size, epochs, alpha, lr):
    """
    For pruning, our initial sparsity is 0.5, our final sparsity is 0.9, the pruning frequency is 10 steps, and we use
    a cubic pruning schedule (see Zhu and Gupta (2017)). Initial and final sparsities were chosen due to their use in
    the TensorFlow Model Optimization Tutorial.
    """
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    (x_train, y_train), (x_test, y_test) = utils.get_flattened_fashion_mnist()

    # number of training examples to use. Must be multiple of batch_size as the current loss cannot deal with batches
    # where the number of examples are not a multiple of batch_size. This should only result in negligible differences
    # to the clusterability paper
    limit = (batch_size * (len(x_train) // batch_size))
    test_limit = (batch_size * (len(x_test) // batch_size))  # Must also be multiple of batch_size

    end_step = np.ceil(limit // batch_size) * epochs

    c0, c1 = [], []
    for c in list(itertools.combinations(range(batch_size), 2)):
        c0.append(c[0])
        c1.append(c[1])

    test_set_accs = []
    for trial in range(1, 6):

        log_dir = '../../experiments/fashion_mnist/alpha_{}/trial_{}/pruned/'.format(alpha, trial)
        saved_model_dir = '../../experiments/fashion_mnist/alpha_{}/trial_{}/'.format(alpha, trial)

        #model, (i, i2, e1, e2, e3, e4, o) = utils.get_mnist_clusterability_model()
        model, _ = utils.get_mnist_clusterability_model()

        model.load_weights(os.path.join(saved_model_dir, 'inference_model.h5'))

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                     final_sparsity=0.90,
                                                                     begin_step=0,
                                                                     frequency=10,
                                                                     end_step=end_step)
        }

        model_for_pruning = prune_low_magnitude(model, **pruning_params)

        model_for_pruning.summary()

        i = model_for_pruning.get_layer('input_1')
        i2 = model_for_pruning.get_layer('input_2')
        e1 = model_for_pruning.get_layer('prune_low_magnitude_dense_1')(i)
        e2 = model_for_pruning.get_layer('prune_low_magnitude_dense_2')(e1)
        e3 = model_for_pruning.get_layer('prune_low_magnitude_dense_3')(e2)
        e4 = model_for_pruning.get_layer('prune_low_magnitude_dense_4')(e3)
        o = model_for_pruning.get_layer('prune_low_magnitude_softmax')(e4)

        model_for_pruning = tf.keras.Model(inputs=[i, i2], outputs=o)

        model_for_pruning.add_loss(utils.get_mnist_clusterability_loss(i2, o, e1, e2, e3, e4,
                                                                       np.array(c0, dtype='int32'),
                                                                       np.array(c1, dtype='int32'),
                                                                       alpha=alpha))
        # clusterability paper uses Adam with lr=0.001
        model_for_pruning.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=None)

        model_for_pruning.add_metric(utils.cce_metric(i2, o), name='cce', aggregation='mean')
        model_for_pruning.add_metric(utils.get_mnist_clusterability_loss(i2, o, e1, e2, e3, e4, c0, c1, alpha),
                         name='p_loss', aggregation='mean')
        model_for_pruning.add_metric(utils.acc_metric(i2, o), name='acc', aggregation='mean')
        model_for_pruning.add_metric(utils.ps_term_metric(i2, e1, c0, c1), name='ps1', aggregation='mean')
        model_for_pruning.add_metric(utils.ps_term_metric(i2, e2, c0, c1), name='ps2', aggregation='mean')
        model_for_pruning.add_metric(utils.ps_term_metric(i2, e3, c0, c1), name='ps3', aggregation='mean')
        model_for_pruning.add_metric(utils.ps_term_metric(i2, e4, c0, c1), name='ps4', aggregation='mean')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=10,
                write_images=True,
                embeddings_freq=10
        )
        history = model_for_pruning.fit([
            np.array(x_train)[:limit], y_train[:limit]],
            y=None,
            validation_data=([np.array(x_test)[:test_limit],
                              y_test[:test_limit]], None),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tensorboard_callback]
        )

        utils.generate_history_plots(history, ['loss', 'cce', 'p_loss', 'acc', 'ps1', 'ps2', 'ps3', 'ps4'],
                                     epochs, save_path=os.path.join(log_dir, 'history.png'))

        # measure test set accuracy
        inference = tf.keras.Model(inputs=i, outputs=o)
        inference.save_weights(os.path.join(log_dir, 'inference_model.h5'))
        y_pred_test = inference.predict(np.array(x_test))
        print('[{}] test set accuracy : {}'.format(trial, np.mean(utils.acc_metric(y_test, y_pred_test))))
        test_set_accs.append(np.mean(utils.acc_metric(y_test, y_pred_test)))

    print('test_set_accs: {}'.format(test_set_accs))
    print('mean test set accuracy across trials: {}'.format(np.mean(test_set_accs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="experiment configuration file (.yml)",
                        type=str, required=True)

    args = parser.parse_args()
    config_file = args.config

    config = yaml.safe_load(open(config_file, 'r'))
    print(config)

    if config['experiment'] == 'mnist':
        run_mnist(config['batch_size'], config['epochs'], config['alpha'], config['lr'])
    elif config['experiment'] == 'mnist_with_pruning':
        run_mnist_with_pruning(config['batch_size'], config['epochs'], config['alpha'], config['lr'])
    elif config['experiment'] == 'fashion_mnist':
        run_fashion_mnist(config['batch_size'], config['epochs'], config['alpha'], config['lr'])
    elif config['experiment'] == 'fashion_mnist_with_pruning':
        run_fashion_mnist_with_pruning(config['batch_size'], config['epochs'], config['alpha'], config['lr'])


if __name__ == '__main__':
    main()
