def mnist_layer_analysis():
    train_ps_term_metrics = []

    # layer 1 analysis
    embed1 = tf.keras.Model(inputs=i, outputs=e1)
    embed1.layers[1].set_weights(model.layers[1].get_weights())

    embed1_activations_by_class, embed1_classwise_ps_term_metric = \
        utils.plot_activations(embed1, np.array(x_train), np.argmax(y_train, axis=1))

    print('[{}] training set embed1 ps_term_metric : {}'.format(trial, embed1_classwise_ps_term_metric))
    train_ps_term_metrics.append(embed1_classwise_ps_term_metric)

    utils.generate_visualise_hidden_neurons_plot(model, 1, 256, (28, 28))

    # layer 2 analysis
    embed2_input = tf.keras.layers.Input(shape=(28 * 28,))
    embed2_hidden1 = tf.keras.layers.Dense(256, activation='relu', name='embed2_hidden1')(embed2_input)
    embed2_output = tf.keras.layers.Dense(256, activation='relu', name='embed2_output')(embed2_hidden1)
    embed2 = tf.keras.Model(inputs=embed2_input, outputs=embed2_output)

    embed2.layers[1].set_weights(model.layers[1].get_weights())
    embed2.layers[2].set_weights(model.layers[2].get_weights())

    embed2_activations_by_class, embed2_classwise_ps_term_metric = \
        utils.plot_activations(embed2, np.array(x_train), np.argmax(y_train, axis=1))

    print('[{}] training set embed2 ps_term_metric : {}'.format(trial, embed2_classwise_ps_term_metric))
    train_ps_term_metrics.append(embed2_classwise_ps_term_metric)

    utils.generate_visualise_hidden_neurons_plot(model, 2, 256, (16, 16))

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
        utils.plot_activations(embed3, np.array(x_train), np.argmax(y_train, axis=1))

    print('[{}] training set embed3 ps_term_metric : {}'.format(trial, embed3_classwise_ps_term_metric))
    train_ps_term_metrics.append(embed2_classwise_ps_term_metric)

    utils.generate_visualise_hidden_neurons_plot(model, 3, 256, (16, 16))

    # layer 4 analysis