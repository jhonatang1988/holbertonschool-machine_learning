#!/usr/bin/env python3
"""
performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a valid convolution on grayscale images
    :param stride: is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    :param padding: is a tuple of (ph, pw)
    ph is the padding for the height of the image
    pw is the padding for the width of the image
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
    sh, sw = stride

    if padding:
        if type(padding) is tuple:
            pad_height, pad_width = padding
        elif padding == 'same':
            pad_height = int(((h * sh + kh - h) / 2) + 1)
            pad_width = int(((w * sw + kw - w) / 2) + 1)

            # if kw % 2 != 0:
            #     pad_width = int((((h - 1) * sh + kh - h) / 2) + 1)
            #
            # if kh % 2 != 0:
            #     pad_height = int((((w - 1) * sw + kw - w) / 2) + 1)
        else:
            pad_height, pad_width = (0, 0)

    padded_images = np.pad(images, ((0, 0), (pad_height, pad_height),
                                    (pad_width, pad_width)), 'constant')

    Y = np.zeros((m,
                  ((h - kh + 2 * pad_height) // sh) + 1,
                  ((w - kw + 2 * pad_width) // sw) + 1))
    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            slc = \
                padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            slc = slc * kernel
            # we have to get a number from result (matrix of matrices)
            slc = slc.sum(axis=1)
            slc = slc.sum(axis=1)
            # once we get the numbers we are going to
            # store the numbers in the same position for
            # each image
            Y[:, i, j] = slc

    return Y
