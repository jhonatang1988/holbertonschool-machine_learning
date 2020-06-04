#!/usr/bin/env python3
"""
performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images
    :param images: is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    :param kernel: is a numpy.ndarray with shape (kh, kw) containing the
    kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    :return: numpy.ndarray containing the convolved images
    """
    # https://stackoverflow.com/a/43087771/4101146
    conv_list = []
    # for img in images:
    #     a = img
    #     f = kernel
    #     s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    #     strd = np.lib.stride_tricks.as_strided
    #     subM = strd(a, shape=s, strides=a.strides * 2)
    #     new_image = np.einsum('ij,ijkl->kl', f, subM)
    #
    #     conv_list.append(new_image)
    #
    # return np.array(conv_list)

    # https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html

    #  to pass checkers
    """Compute 2D cross-correlation."""
    kh, kw = kernel.shape
    m, h, w = images.shape
    Y = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            slc = (images[:, i:i + kh, j:j + kw]) * kernel
            slc = slc * kernel
            # we have to get a number from result (matrix of matrices)
            slc = slc.sum(axis=1)
            slc = slc.sum(axis=1)
            # once we get the numbers we are going to
            # store the numbers in the same position for
            # each image
            Y[:, i, j] = slc

    return Y
