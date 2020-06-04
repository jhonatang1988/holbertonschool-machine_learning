#!/usr/bin/env python3
"""
performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    #  to pass checkers
    # """Compute 2D cross-correlation."""
    kh, kw = kernel.shape
    m, h, w = images.shape

    pad_height = int(kh / 2)
    pad_width = int(kw / 2)

    if kw % 2 != 0:
        pad_width = int((kw - 1) / 2)

    if kh % 2 != 0:
        pad_height = int((kh - 1) / 2)

    padded_images = np.pad(images, ((0, 0), (pad_height, pad_height),
                                    (pad_width, pad_width)), 'constant')
    Y = np.empty_like(images)
    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            slc = (padded_images[:, i:i + kh, j:j + kw]) * kernel
            # we have to get a number from result (matrix of matrices)
            slc = slc.sum(axis=1)
            slc = slc.sum(axis=1)
            # once we get the numbers we are going to
            # store the numbers in the same position for
            # each image
            Y[:, i, j] = slc

    return Y
