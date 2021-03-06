#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent, also analyze validation data,
and has early_stopping
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent
    :param patience: the patience used for early stopping
    :param early_stopping: is a boolean that indicates whether early stopping
    should be used
    :param validation_data: the data to validate the model with, if not None
    :param network: model to train
    :param data: numpy.ndarray of shape (m, nx) containing the input data
    :param labels: one-hot numpy.ndarray of shape (m, classes)
    containing the labels of data
    :param batch_size: is the size of the batch used for mini-batch gradient
    descent
    :param epochs: is the number of passes through data for mini-batch gradient
    descent
    :param verbose: is a boolean that determines if output should be printed
    during training
    :param shuffle: is a boolean that determines whether to shuffle the batches
    every epoch. Normally, it is a good idea to shuffle, but for
    reproducibility, we have chosen to set the default to False
    :return: the History object generated after training the model
    """
    # https://www.tensorflow.org/guide/keras/train_and_evaluate
    # the callbacks

    if early_stopping is True and validation_data:
        callbacks = [
            K.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                monitor='val_loss',
                patience=patience
            )
        ]
    else:
        callbacks = None

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)

    return history
