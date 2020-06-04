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
    # https://stackoverflow.com/a/22341269/4101146
    conv_list = []
    for image in images:
        x = image
        y = kernel
        x_shape = np.array(x.shape)
        y_shape = np.array(y.shape)
        z_shape = x_shape + y_shape - 1
        z = np.fft.ifft2(
            np.fft.fft2(x, z_shape) * np.fft.fft2(y, z_shape)).real

        # To compute a valid shape, either np.all(x_shape >= y_shape) or
        # np.all(y_shape >= x_shape).
        valid_shape = x_shape - y_shape + 1
        if np.any(valid_shape < 1):
            valid_shape = y_shape - x_shape + 1
            if np.any(valid_shape < 1):
                raise ValueError("empty result for valid shape")
        start = (z_shape - valid_shape) // 2
        end = start + valid_shape
        z = z[start[0]:end[0], start[1]:end[1]]
        conv_list.append(z)

    return np.asarray(conv_list)
