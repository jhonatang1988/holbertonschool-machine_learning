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
    for img in images:
        a = img
        f = kernel
        s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
        strd = np.lib.stride_tricks.as_strided
        subM = strd(a, shape=s, strides=a.strides * 2)

        conv_list.append(np.einsum('ij,ijkl->kl', f, subM))

    return np.asarray(conv_list)
