#!/usr/bin/env python3
"""
that trains a loaded neural network model using mini-batch gradient descent:
"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data

"""
returns a mini batch
"""


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


"""
that trains a loaded neural network model using mini-batch gradient descent:
"""


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    that trains a loaded neural network model using mini-batch gradient
    descent:
    :param X_train: (m, 784) training data
    :param Y_train: (m, 10) labels
    :param X_valid: (m, 784) containing the validation data
    :param Y_valid: (m, 10) containing the validation labels
    :param batch_size: number of data points in a batch
    :param epochs: number of times the training should pass through
        the whole dataset
    :param load_path: path from which to load the model
    :param save_path: path to where the model should be saved after training
    :return: the path where the model was saved
    """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
            train_cost = sess.run(loss,
                                  feed_dict={x: X_train,
                                             y: Y_train})
            train_accuracy = sess.run(accuracy,
                                      feed_dict={x: X_train,
                                                 y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid,
                                                   y: Y_valid})
            valid_accuracy = sess.run(accuracy,
                                      feed_dict={x: X_valid,
                                                 y: Y_valid})
            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))

            if epoch < epochs:

                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
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

        saved_model = loader.save(sess, save_path)
    return saved_model
