#!/usr/bin/env python3
"""
builds a modified version of the LeNet-5 architecture using tensorflow
"""

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
    # el initializer 3000
    init = tf.contrib.layers.variance_scaling_initializer()
    
    # la mierda que recibe inputs
    primer_layer = tf.layers.Conv2D(filters=6, kernel_size=[5, 5],
                                    padding='same', kernel_initializer=init,
                                    activation='relu')
    
    x = primer_layer(x)
    
    # ahora vamos con un max poolincibiribiri
    segundo_layer = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                           strides=(2, 2))
    
    x = segundo_layer(x)
    
    # ahora vamos con otra puta convolutional
    tercer_layer = tf.layers.Conv2D(filters=16, kernel_size=[5, 5],
                                    padding='valid',
                                    kernel_initializer=init,
                                    activation='relu')
    
    x = tercer_layer(x)
    
    # ahora vamos con otra piscina la mas chimba
    cuarto_layer = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                          strides=(2, 2))
    
    x = cuarto_layer(x)
    
    # esta mierda es para pasar las imagenes a 1D porque la vaina no
    # funciona en otras dimensions
    quinto_layer = tf.layers.Flatten()
    
    x = quinto_layer(x)
    
    # ahora vamos con la vieja confiable de un fully connected - orgia total.
    sexto_layer = tf.layers.Dense(units=120, activation='relu',
                                  kernel_initializer=init)
    
    x = sexto_layer(x)
    
    # otra vieja confiable
    septimo_layer = tf.layers.Dense(units=84, activation='relu',
                                    kernel_initializer=init)
    
    x = septimo_layer(x)
    
    # una ultima softmax
    ultimo_layer = tf.layers.Dense(units=10, activation='softmax')
    
    x = ultimo_layer(x)
    
    # toca definir el loss para retornarlo
    loss = tf.losses.softmax_cross_entropy(y, x)
    
    # como se optimiza la enseñanza del bb
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    # el accuracy
    pred = tf.argmax(x, 1)
    eq = tf.equal(pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))
    
    return x, optimizer, loss, accuracy
