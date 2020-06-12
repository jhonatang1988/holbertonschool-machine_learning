#!/usr/bin/env python3
"""
builds a modified version of the LeNet-5 architecture using tensorflow
"""

import numpy as np
import tensorflow as tf


# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
# tf.keras.layers.Conv2D(
#  filters,
#  kernel_size,
#  strides=(1, 1),
#  padding='valid',
#  data_format=None,
#  dilation_rate=(1, 1),
#  activation=None,
#  use_bias=True,
#  kernel_initializer='glorot_uniform',
#  bias_initializer='zeros',
#  kernel_regularizer=None,
#  bias_regularizer=None,
#  activity_regularizer=None,
#  kernel_constraint=None,bias_constraint=None,
#  **kwargs )

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
# tf.keras.layers.MaxPool2D(
# pool_size=(2, 2),
# strides=None,
# padding='valid',
# data_format=None,
# **kwargs )

# tf.keras.layers.Dense(
# units,
# activation=None,
# use_bias=True,
# kernel_initializer='glorot_uniform',
# bias_initializer='zeros',
# kernel_regularizer=None,
# bias_regularizer=None,
# activity_regularizer=None,
# kernel_constraint=None,
# bias_constraint=None,
# **kwargs )

# tf.keras.layers.Softmax(
# axis=-1,
# **kwargs )

def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using tensorflow
    :param x: x is a tf.placeholder of shape (m, 28, 28, 1) containing the
    input images for the network
    m is the number of images
    :param y: y is a tf.placeholder of shape (m, 10) containing the one-hot
    labels for the network
    :return:
    """
    
    # se crea un graph con ordenes secuenciales. Se utiliza
    # tf.models.Sequential()
    model = tf.models.Sequential()
    
    # ahora se crea el primer layer convolutional que tiene lo siguiente:
    # filtros (entero) - size del kernel o filtro, que debe tener altura y
    # largo en una tupla o lista
    model.add(tf.layers.Conv2D(filters=6, kernel_size=[5, 5],
                               padding='same'))
    
    # ahora vamos con un max poolincibiribiri
    model.add(tf.layers.MaxPooling2D(pool_size=(2, 2),
                                     strides=(2, 2)))
    # ahora vamos con otra puta convolutional
    model.add(tf.layers.Conv2D(filters=16, kernel_size=[5, 5],
                               padding='valid'))
    
    # ahora vamos con otra piscina la mas chimba
    model.add(tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # esta mierda es para pasar las imagenes a 1D porque la vaina no
    # funciona en otras dimensions
    model.add(tf.layers.Flatten())
    
    # ahora vamos con la vieja confiable de un fully connected - orgia total.
    model.add(tf.layers.Dense(units=120, activation='relu'))
    
    # otra vieja confiable
    model.add(tf.layers.Dense(units=84, activation='relu'))
    
    # una ultima softmax
    softmax = model.add(tf.layers.Dense(units=10, activation='softmax'))
    
    # toca definir el loss para retornarlo
    loss = tf.losses.softmax_cross_entropy()
    
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    # se compila la vuelta
    model.compile(optimizer='adam', )
    
    # el accuracy
    accuracy, _ = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                      predictions=tf.argmax(x, 1))
    
    return softmax, optimizer, loss, accuracy
