import os
import itertools
import matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from tqdm import tqdm


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

colour_scheme = [
    '#FF9000',
    '#60D394',
    '#537D8D',
    '#A4031F',
    '#240B36'
]

def calculate_ps_term(e, c0, c1):
    a = tf.gather(indices=c0, params=e) \
      / tf.tile(K.expand_dims(K.max(K.abs(tf.gather(indices=c0, params=e)), axis=1) + 1e-9), (1, e.shape[-1]))
    b = tf.gather(indices=c1, params=e) \
      / tf.tile(K.expand_dims(K.max(K.abs(tf.gather(indices=c1, params=e)), axis=1) + 1e-9), (1, e.shape[-1]))
    return K.sum(a*b, axis=1)


def acc_metric(y_true, y_pred):
    y_pred = tf.cast(tf.math.greater(y_pred, tf.constant([0.5])), dtype='float32')
    return tf.cast(tf.math.equal(y_true, y_pred), dtype='float32')


def cce_metric(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    )(y_true=y_true, y_pred=y_pred)


def ps_term_metric(y_true, e, c0, c1):
    p_loss = (1. - (tf.cast(tf.equal(K.argmax(tf.gather(indices=c0, params=y_true)),
                                     K.argmax(tf.gather(indices=c1, params=y_true))), dtype='float32'))) \
           * calculate_ps_term(e, c0, c1)
    return tf.convert_to_tensor(p_loss)


def generate_history_plots(history, metrics, epochs, save_path=None):
    fig = plt.figure(figsize=(10, 10*len(metrics)))
    for j, m in enumerate(metrics):
        ax = fig.add_subplot(8, 1, j + 1)
        ax.plot(range(1, epochs + 1), np.squeeze(history.history[m]), 'k-', lw=4)
        t, = ax.plot(range(1, epochs + 1), np.squeeze(history.history[m]), '-', color='#7E3F8F', lw=3)
        ax.plot(range(1, epochs + 1), np.squeeze(history.history['val_{}'.format(m)]), 'k-', lw=4)
        v, = ax.plot(range(1, epochs + 1), np.squeeze(history.history['val_{}'.format(m)]), '-', color='#3BCEAC', lw=3)
        ax.set_ylabel(m)
        ax.set_xlabel('epoch')
        if j == 0:
            ax.legend([t, v], ['train', 'val'])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')

def generate_activations_plot(embed, x, y, save_path=None):
    activations_by_class = {}
    a = embed.predict(x)
    for j in tqdm(range(len(x))):
        try:
            activations_by_class[y[j]] += np.squeeze(np.array(a[j] > 0., dtype='uint8'))
        except KeyError:
            activations_by_class[y[j]] = np.squeeze(np.array(a[j] > 0., dtype='uint8'))
    fig = plt.figure(figsize=(12,8))
    classes = list(activations_by_class.keys())
    classes.sort()
    for i,k in enumerate(classes):
        y_pos = np.arange(len(activations_by_class[k]))
        ax = fig.add_subplot(2, 5, i+1)
        ax.set_title(k)
        ax.barh(y_pos, np.squeeze(activations_by_class[k]), align='center')
        ax.plot([0,0],[-1,len(np.squeeze(activations_by_class[k]))], 'k-')
        ax.set_ylim(-1,len(np.squeeze(activations_by_class[k])))
        fig.text(0.5, 0.04, 'frequency neuron activated', ha='center')
        fig.text(0.04, 0.5, 'neuron index', va='center', rotation='vertical')
    sum = 0
    for c in list(itertools.combinations(range(10), 2)):
        a = (np.squeeze(activations_by_class[c[0]]) / (np.max(np.abs(np.squeeze(activations_by_class[c[0]])))) + 1e-9)
        b = (np.squeeze(activations_by_class[c[1]]) / (np.max(np.abs(np.squeeze(activations_by_class[c[1]])))) + 1e-9)
        sum += np.dot(np.transpose(a), b)
    plt.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')
    return activations_by_class, (sum/len(list(itertools.combinations(range(10), 2))))


# TODO: for layers deeper than the first hidden layer the neurons could be visualised as a weighted sum of the first layer activations
def generate_visualise_hidden_neurons_plot(model, layer_index, n_hidden_neurons, vis_shape, save_path=None):
    W = model.layers[layer_index].get_weights()
    fig = plt.figure(figsize=(10,10))
    dim = int(np.ceil(np.sqrt(n_hidden_neurons)))
    for j in range(n_hidden_neurons):
        x_j = W[0][:,j] / np.sqrt(np.sum(np.dot(W[0][:,j], W[0][:,j].T)))
        ax = fig.add_subplot(dim,dim,j+1)
        ax.imshow(x_j.reshape(vis_shape))
        plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')
'''
A.2 Training Details
During training, we use the Adam algorithm (Kingma and Ba 2014) with the standard Keras hyperparameters: learning rate
0.001, β1 = 0.9, β2 = 0.999, no amsgrad. The loss function was categorical cross-entropy except for networks trained on
polynomial regression (see appendix A.7, where we used mean squared error). For pruning, our initial sparsity is 0.5, our final
sparsity is 0.9, the pruning frequency is 10 steps, and we use a cubic pruning schedule (see Zhu and Gupta (2017)). Initial and
final sparsities were chosen due to their use in the TensorFlow Model Optimization Tutorial.7 A batch size of 128 was used for
the MLPs and VGGs, while a batch size of 64 was used for the small CNNs. For MLPs, training went for 20 epochs before
pruning and 20 epochs of pruning. For small CNNs, it was 10 epochs before pruning and 10 epochs of pruning. For VGGs
trained on CIFAR-10, it was 200 epochs before pruning and 50 epochs of pruning. For networks trained on 7×7 MNIST to test
clusterability regularization, it was 10 epochs before pruning and 10 epochs of pruning. The small CNNs trained on MNIST,
Fashion-MNIST, and the stack datasets have all convolutional kernels being 3 by 3, with the second and third hidden layers
being followed by max pooling with a 2 by 2 window. For training the VGG on CIFAR-10, data augmentations used are random
rotations between 0 and 15 degrees, random shifts both vertically and horizontally of up to 10% of the side length, and random
horizontal flipping. We use Tensorflow’s implementation of the Keras API (Abadi et al. 2015; Chollet et al. 2015).
'''

### MNIST methods ###

def get_flattened_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    y_train = np.argmax(y_train, axis=1)

    order = np.random.permutation(len(x_train))

    x_train = x_train[order]
    y_train = y_train[order]

    y_train = tf.keras.utils.to_categorical(y_train, 10)

    x_train_reshape = [x_train[i].reshape((784)) for i in range(len(x_train))]

    x_test_reshape = [x_test[i].reshape((784)) for i in range(len(x_test))]

    return (x_train_reshape, y_train), (x_test_reshape, y_test)


def get_mnist_clusterability_model():
    i = tf.keras.layers.Input(shape=(28 * 28,))
    i2 = tf.keras.layers.Input(shape=(10,))
    e1 = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(i)
    e2 = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(e1)
    e3 = tf.keras.layers.Dense(256, activation='relu', name='dense_3')(e2)
    e4 = tf.keras.layers.Dense(256, activation='relu', name='dense_4')(e3)
    o = tf.keras.layers.Dense(10, activation='softmax', name='softmax')(e4)
    model = tf.keras.Model(inputs=[i, i2], outputs=o)
    return model, (i, i2, e1, e2, e3, e4, o)


def get_mnist_clusterability_loss(y_true, y_pred, e1, e2, e3, e4, c0, c1, alpha):
    p_loss = K.sum((1. - (tf.cast(tf.equal(K.argmax(tf.gather(indices=c0, params=y_true)),
                                           K.argmax(tf.gather(indices=c1, params=y_true))), dtype='float32'))) \
                   * calculate_ps_term(e1, c0, c1))

    p_loss += K.sum((1. - (tf.cast(tf.equal(K.argmax(tf.gather(indices=c0, params=y_true)),
                                            K.argmax(tf.gather(indices=c1, params=y_true))), dtype='float32'))) \
                    * calculate_ps_term(e2, c0, c1))

    p_loss += K.sum((1. - (tf.cast(tf.equal(K.argmax(tf.gather(indices=c0, params=y_true)),
                                            K.argmax(tf.gather(indices=c1, params=y_true))), dtype='float32'))) \
                    * calculate_ps_term(e3, c0, c1))

    p_loss += K.sum((1. - (tf.cast(tf.equal(K.argmax(tf.gather(indices=c0, params=y_true)),
                                            K.argmax(tf.gather(indices=c1, params=y_true))), dtype='float32'))) \
                    * calculate_ps_term(e4, c0, c1))

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true=y_true,
                                                                      y_pred=y_pred)
    loss += alpha * (p_loss / len(c0))

    return loss


### Fashion MNIST methods ###

def get_flattened_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    y_train = np.argmax(y_train, axis=1)

    order = np.random.permutation(len(x_train))

    x_train = x_train[order]
    y_train = y_train[order]

    y_train = tf.keras.utils.to_categorical(y_train, 10)

    x_train_reshape = [x_train[i].reshape((784)) for i in range(len(x_train))]

    x_test_reshape = [x_test[i].reshape((784)) for i in range(len(x_test))]

    return (x_train_reshape, y_train), (x_test_reshape, y_test)
