#!/usr/bin/env python3
"""
evaluates the output of a neural network
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    evaluate
    X: input data
    Y: one-hot labels for X
    save_path: saved model file
    """
    # https://docs.w3cub.com/tensorflow~python/meta_graph/
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        # tf.add_to_collection('x', x)
        # tf.add_to_collection('y', y)
        # y_pred = forward_prop(x, layer_sizes, activations)
        # # print(y_pred)
        # tf.add_to_collection('y_pred', y_pred)
        # accuracy = calculate_accuracy(y, y_pred)
        # # print(accuracy)
        # tf.add_to_collection('accuracy', accuracy)
        # loss = calculate_loss(y, y_pred)
        # # print(loss)
        # tf.add_to_collection('loss', loss)
        # train_op = create_train_op(loss, alpha)
        # # print(train_op)
        # tf.add_to_collection('train_op', train_op)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Returns: the network’s prediction, accuracy, and loss, respectively
        eval_y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        eval_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        eval_loss = sess.run(loss, feed_dict={x: X, y: Y})

    return eval_y_pred, eval_accuracy, eval_loss
