# #!/usr/bin/env python3
# """
# builds, trains, and saves a neural network classifier
# """
# import tensorflow as tf
# create_placeholders = __import__('0-create_placeholders').create_placeholders
# forward_prop = __import__('2-forward_prop').forward_prop
# calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
# calculate_loss = __import__('4-calculate_loss').calculate_loss
# create_train_op = __import__('5-create_train_op').create_train_op


# def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
#           iterations, save_path="/tmp/model.ckpt"):
#     """
#     builds, trains, and saves a neural network classifier
#     X_train: param: is a numpy.ndarray containing the training input data
#     Y_train: param: is a numpy.ndarray containing the training labels
#     X_valid: param: is a numpy.ndarray containing the validation input data
#     Y_valid: param: is a numpy.ndarray containing the validation labels
#     layer_sizes: param: is a list containing the number of nodes in each \
#         layer of the network
#     activations: param: is a list containing the activation functions for \
#         each layer of the network
#     alpha: param: is the learning rate
#     iterations: param: is the number of iterations to train over
#     save_path: param: designates where to save the model
#     """
#     # print(X_train.shape[1])
#     # print(Y_train.shape)
#     # print(X_valid.shape)
#     # print(Y_valid.shape)
#     # print(layer_sizes)
#     # print(activations)

#     x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
#     # print(x)
#     # print(y)
#     tf.add_to_collection('x', x)
#     tf.add_to_collection('y', x)

#     y_pred = forward_prop(x, layer_sizes, activations)
#     # print(y_pred)
#     tf.add_to_collection('y_pred', y_pred)

#     accuracy = calculate_accuracy(y, y_pred)
#     # print(accuracy)
#     tf.add_to_collection('accuracy', accuracy)

#     loss = calculate_loss(y, y_pred)
#     # print(loss)
#     tf.add_to_collection('loss', loss)

#     train_op = create_train_op(loss, alpha)
#     # print(train_op)
#     tf.add_to_collection('train_op', train_op)

#     init_op = tf.global_variables_initializer()
#     # print(init_op)
#     saver = tf.train.Saver()

#     init_op = tf.global_variables_initializer()
#     saver = tf.train.Saver()

#     with tf.Session() as sess:
#         sess.run(init_op)
#         for i in range(0, iterations + 1, 1):
#             cost_train = sess.run(
#                 loss, feed_dict={x: X_train, y: Y_train})
#             acc_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
#             cost_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
#             acc_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
#             print('After {} iterations:'.format(i)
#                   ) if i % 100 == 0 or i == iterations else None
#             print('\tTraining Cost: {}'.format(
#                 cost_train)) if i % 100 == 0 or i == iterations else None
#             print('\tTraining Accuracy: {}'.format(
#                 acc_train)) if i % 100 == 0 or i == iterations else None
#             print('\tValidation Cost: {}'.format(
#                 cost_valid)) if i % 100 == 0 or i == iterations else None
#             print('\tValidation Accuracy: {}'.format(
#                 acc_valid)) if i % 100 == 0 or i == iterations else None

#             sess.run(train_op, feed_dict={
#                      x: X_train, y: Y_train}) if i < iterations else None

#     saved_model = saver.save(sess, save_path)

#     return saved_model
#!/usr/bin/env python3
"""
Train with Iterations tensorflow
"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):
    """
    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes is a list containing the number of nodes
                in each layer of the network
    actications is a list containing the activation functions
                for each layer of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as ses:
        ses.run(init)
        for i in range(iterations + 1):
            cost_train = ses.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_train = ses.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_valid = ses.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_valid = ses.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(acc_valid))

            if i < iterations:
                ses.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(ses, save_path)
    return save_path
