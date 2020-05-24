#!/usr/bin/env python3
"""
builds, trains, and saves a neural network model in tensorflow using Adam
optimization, mini-batch gradient descent, learning rate decay, and batch
normalization
"""
import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices
    :param X: first matrix
    :param Y: second matrix
    :return: both matrices shuffled
    """
    # https://stackoverflow.com/questions/43229034/randomly-shuffle-data-
    # and-labels-from-different-files-in-the-same-order
    idx = np.random.permutation(Y.shape[0])
    return X[idx], Y[idx]


def iterate_mini_batches(X, Y, batch_size):
    """
    to get the mini_batch
    :param X: input data
    :param Y: labels
    :param batch_size: size of mini batch
    :return: mini_batch
    """
    # https://stackoverflow.com/questions/38157972/how-to-implement-mini-
    # batch-gradient-descent-in-python SECOND ANSWER!
    j = 0
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X.shape[0])
        j = j + 1
        excerpt = slice(start_idx, end_idx)
        yield X[excerpt], Y[excerpt], j


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction
    y: param: prediction
    y_pred: param: prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow
    :param prev: activated output of the previous layer
    :param n: number of nodes in the layer to be created
    :param activation: activation function that should be used on the output
    of the layer
    :return:
    """
    # https://stackoverflow.com/questions/33949786/how-could-i-use-batch
    # -normalization-in-tensorflow
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n,
                             name='layer', kernel_initializer=init)

    Z = output(prev)
    mean, variance = tf.nn.moments(Z, [0])

    epsilon = 1e-8

    beta = tf.Variable(tf.constant(0.0, shape=[n]))
    gamma = tf.Variable(tf.constant(1.0, shape=[n]))
    Z_norm = tf.nn.batch_normalization(
        x=Z, mean=mean, variance=variance, offset=beta, scale=gamma,
        variance_epsilon=epsilon)

    if not activation:
        return Z_norm
    else:
        return activation(Z_norm)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    x: placeholder for the input data
    layer_sizes: a list containing # of nodes in the layer
    activations: a list containing activations
    """
    for i in range(len(layer_sizes)):
        if i == 0:
            output = create_batch_norm_layer(x, layer_sizes[i], activations[i])
        else:
            output = create_batch_norm_layer(output, layer_sizes[i],
                                             activations[i])
    return output


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation in tensorflow using inverse time
    decay
    :param alpha: learning rate
    :param decay_rate: weight used to determine the rate at which alpha will
    decay
    :param global_step: is the number of passes of gradient descent that have
    elapsed
    :param decay_step:  is the number of passes of gradient descent that should
    occur before alpha is decayed further
    :return: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       learning_rate=alpha,
                                       staircase=True)


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    y: param: prediction
    y_pred: param: prediction
    returns: tensor with the accuracy
    """
    # https://stackoverflow.com/questions/42607930/how-to-compute-accuracy-of-cnn-in-tensorflow
    pred = tf.argmax(y_pred, 1)
    eq = tf.equal(pred, tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(eq, tf.float32))

    return acc


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta1: weight used for the first moment
    :param beta2: weight used for the second moment
    :param epsilon: small number to avoid ZeroDivisionError
    :return: Adam optimization operation
    """
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                  beta2=beta2, epsilon=epsilon).minimize(loss)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    builds, trains, and saves a neural network model in tensorflow
    :param Data_train: tuple containing the training inputs and training
    labels, respectively
    :param Data_valid: tuple containing the validation inputs and validation
    labels, respectively
    :param layers: list containing the number of nodes in each layer of the
    network
    :param activations: list containing the activation functions used for each
    layer of the network
    :param alpha: the learning rate
    :param beta1: the weight for the first moment of Adam Optimization
    :param beta2: the weight for the second moment of Adam Optimization
    :param epsilon: small number used to avoid division by zero
    :param decay_rate: is the decay rate for inverse time decay of the
    learning rate (the corresponding decay step should be 1)
    :param batch_size: is the number of data points that should be in a
    mini-batch
    :param epochs: the number of times the training should pass through the
    whole dataset
    :param save_path: the path where the model should be saved to
    :return: path where the model was saved
    """

    """first the placeholders"""
    x = tf.placeholder(tf.float32, shape=[None, Data_train[0].shape[1]],
                       name='x')
    y = tf.placeholder(tf.float32, shape=[None, Data_train[1].shape[1]],
                       name='y')

    y_pred = forward_prop(x, layers, activations)
    loss = calculate_loss(y, y_pred)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    accuracy = calculate_accuracy(y, y_pred)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    # lets save the graph
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for epoch in range(epochs + 1):
            train_cost = sess.run(
                loss, feed_dict={x: Data_train[0], y: Data_train[1]})
            train_accuracy = sess.run(accuracy, feed_dict={x: Data_train[0],
                                                           y: Data_train[1]})
            valid_cost = sess.run(loss, feed_dict={x: Data_valid[0],
                                                   y: Data_valid[1]})
            valid_accuracy = sess.run(accuracy,
                                      feed_dict={x: Data_valid[0],
                                                 y: Data_valid[1]})

            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))

            if epoch < epochs:

                X_shuffled, Y_shuffled = shuffle_data(Data_train[0],
                                                      Data_train[1])
                sess.run(alpha)
                sess.run(global_step.assign(epoch))
                for batch in iterate_mini_batches(X_shuffled, Y_shuffled,
                                                  batch_size):
                    x_batch, y_batch, j = batch
                    sess.run(train_op, feed_dict={x: x_batch, y: y_batch})
                    if j % 100 == 0 and j != 0:
                        step_cost = sess.run(loss,
                                             feed_dict={x: x_batch,
                                                        y: y_batch})
                        step_accuracy = sess.run(accuracy,
                                                 feed_dict={x: x_batch,
                                                            y: y_batch})
                        print('\tStep {}:'.format(j))
                        print('\t\tCost: {}'.format(step_cost))
                        print('\t\tAccuracy: {}'.format(step_accuracy))

        saved_model = saver.save(sess, save_path)
    return saved_model
