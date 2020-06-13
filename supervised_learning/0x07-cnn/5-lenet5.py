#!/usr/bin/env python3
"""
builds a modified version of the LeNet-5 architecture using keras
"""

import tensorflow.keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5 architecture using keras
    :param X: X is a K.Input of shape (m, 28, 28, 1) containing the input
    images for the network
    m is the number of images
    :return: a K.Model compiled to use Adam optimization (with default
    hyperparameters) and accuracy metrics
    """
    
    init = K.initializers.he_normal()
    primer_layer = K.layers.Conv2D(
        filters=6,
        kernel_size=[5, 5],
        kernel_initializer=init,
        activation='relu',
        padding='same'
    )
    
    Y_pred = primer_layer(X)
    
    segundo_layer = K.layers.MaxPooling2D(pool_size=(2, 2),
                                          strides=(2, 2))
    
    Y_pred = segundo_layer(Y_pred)
    
    tercer_layer = K.layers.Conv2D(filters=16,
                                   kernel_size=[5, 5],
                                   kernel_initializer=init,
                                   activation='relu'
                                   )
    
    Y_pred = tercer_layer(Y_pred)
    
    cuarto_layer = K.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2))
    
    Y_pred = cuarto_layer(Y_pred)
    
    quinto_layer = K.layers.Flatten()
    
    Y_pred = quinto_layer(Y_pred)
    
    sexto_layer = K.layers.Dense(units=120,
                                 kernel_initializer=init,
                                 activation='relu')
    
    Y_pred = sexto_layer(Y_pred)
    
    septimo_layer = K.layers.Dense(units=84,
                                   kernel_initializer=init,
                                   activation='relu')
    
    Y_pred = septimo_layer(Y_pred)
    
    ultimo_layer_con_activacion = K.layers.Dense(units=10,
                                                 kernel_initializer=init,
                                                 activation='softmax')
    
    Y_pred = ultimo_layer_con_activacion(Y_pred)
    
    model = K.models.Model(inputs=X, outputs=Y_pred)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    
    return model
