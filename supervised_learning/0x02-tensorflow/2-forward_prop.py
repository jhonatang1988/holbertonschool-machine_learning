#!/usr/bin/env python3
"""
creates the forward propagation graph for the neural network
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    x: placeholder for the input data
    layer_sizes: a list containing # of nodes in the layer
    activations: a list containing activations
    """
    for i in range(len(layer_sizes)):
        if i == 0:
            output = create_layer(x, layer_sizes[i], activations[i])
        else:
            output = create_layer(output, layer_sizes[i], activations[i])
    return output
