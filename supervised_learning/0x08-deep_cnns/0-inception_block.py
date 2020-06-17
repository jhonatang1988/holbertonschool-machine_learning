#!/usr/bin/env python3
"""
that builds an inception block as described in Going Deeper with Convolutions
https://arxiv.org/pdf/1409.4842.pdf
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    that builds an inception block as described in Going Deeper with
    Convolutions
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
    F1 is the number of filters in the 1x1 convolution
    F3R is the number of filters in the 1x1 convolution before the 3x3
    convolution
    F3 is the number of filters in the 3x3 convolution
    F5R is the number of filters in the 1x1 convolution before the 5x5
    convolution
    F5 is the number of filters in the 5x5 convolution
    FPP is the number of filters in the 1x1 convolution after the max pooling
    :return: the concatenated output of the inception block
    """
    init = K.initializers.he_normal()
    F1, F3R, F3, F5R, F5, FPP = filters
    
    # esta convolution va directo entre inputs y outputs
    non_linear_1_1_A = K.layers.Conv2D(
        filters=F1,
        kernel_size=[1, 1],
        kernel_initializer=init,
        activation='relu',
        padding='same'
    )
    to_concatenate_A = non_linear_1_1_A(A_prev)
    
    # esta convolution coge inputs pero va a otra convolution de 3x3
    non_linear_1_1_B = K.layers.Conv2D(
        filters=F3R,
        kernel_size=[1, 1],
        activation='relu'
    )
    
    to_conv_A = non_linear_1_1_B(A_prev)
    
    # esta convolution coge inputs pero va a otra convolution de 5x5
    non_linear_1_1_C = K.layers.Conv2D(
        filters=F5R,
        kernel_size=[1, 1],
        activation='relu'
    )
    
    to_conv_B = non_linear_1_1_C(A_prev)
    
    # esta max pooling coge inputs pero va a una convolution de 1x1
    linear_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )
    
    to_conv_C = linear_pool(A_prev)
    
    # esta convolution coge el input de la 1x1 convolution
    non_linear_1_1_E = K.layers.Conv2D(
        filters=F3,
        kernel_size=[3, 3],
        activation='relu',
        padding='same'
    )
    
    to_concatenate_B = non_linear_1_1_E(to_conv_A)
    
    non_linear_1_1_F = K.layers.Conv2D(
        filters=F5,
        kernel_size=[5, 5],
        activation='relu',
        padding='same'
    )
    
    to_concatenate_C = non_linear_1_1_F(to_conv_B)
    
    non_linear_1_1_G = K.layers.Conv2D(
        filters=FPP,
        kernel_size=[1, 1],
        activation='relu',
        padding='same'
    )
    
    to_concatenate_D = non_linear_1_1_G(to_conv_C)
    
    concatenate_all = K.layers.concatenate([to_concatenate_A,
                                            to_concatenate_B,
                                            to_concatenate_C,
                                            to_concatenate_D])
    
    return concatenate_all
