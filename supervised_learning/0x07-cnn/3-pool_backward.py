#!/usr/bin/env python3
"""
performs back propagation over a pooling layer of a neural network.
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
     output of the previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
    for the pooling
    kh is the kernel height
    kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height
    sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    you may import numpy as np
    Returns: the partial derivatives with respect to the previous layer (
    dA_prev)
    """
    H_P, W_P = kernel_shape
    SH, SW = stride

    dx = np.zeros(A_prev.shape)

    N, HH, WW, C = dA.shape

    for n in range(N):
        for depth in range(C):
            for r in range(HH):
                for c in range(WW):
                    x_pool = A_prev[
                             n, r * SH: r * SH + H_P,
                             c * SW: c * SW + W_P, depth]

                    if mode == 'max':
                        mask = (x_pool == np.max(x_pool))
                        nocabe = dx[
                                 n, r * SH:r * SH + H_P,
                                 c * SW: c * SW + W_P, depth]
                        nocabe += mask * dA[n, r, c, depth]

                    if mode == 'avg':
                        mask = np.ones(x_pool.shape)
                        avg = dA[n, r, c, depth] / (H_P * W_P)
                        nocabe = dx[
                                 n, r * SH:r * SH + H_P,
                                 c * SW: c * SW + W_P, depth]

                        nocabe += mask * avg
    return dx
