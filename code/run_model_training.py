import os
import sys
import yaml
import argparse
import itertools

import numpy as np
import tensorflow as tf

sys.path.insert(0, '../utils')
import utils


def run_mnist(batch_size, epochs, alpha, lr):

    x_train, y_train, x_test, y_test = utils.get_flattened_mnist()

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

        logdir = '../../experiments/mnist/alpha_{}/trial_{}/'.format(alpha, trial)

        model, i, i2, e1, e2, e3, e4, o = utils.get_mnist_clusterability_model()

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
                logdir=logdir,
                histogram_freq=10,
                write_images=True,
                profile_batch=(10, 20),
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
                                     epochs, save_path=os.path.join(logdir, 'history.png'))

        # measure test set accuracy
        inference = tf.keras.Model(inputs=i, outputs=o)
        y_pred_test = inference.predict(np.array(x_test))
        print('[{}] test set accuracy : {}'.format(trial, np.mean(utils.acc_metric(y_test, y_pred_test))))
        test_set_accs.append(np.mean(utils.acc_metric(y_test, y_pred_test)))

    print('test_set_accs: {}'.format(test_set_accs))
    print('mean test set accuracy across trials: {}'.format(np.mean(test_set_accs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="experiment configuration file (.yml)",
                        type=str, required=True)

    args = parser.args()
    config_file = args.config

    config = yaml.load(open(config_file, 'r'))

    if config['experiment'] == 'mnist':
        run_mnist(config['batch_size'], config['epochs'], config['alpha'], config['lr'])


if __name__ == '__main__':
    main()
