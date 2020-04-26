#!/usr/bin/env python3
"""
derivatives for polys
[5, 3, 0, 1]
"""


# poly = [-5, 0, 0, 0, 2, 0, 0, 1]
# = [0, 0, 0, 8, 0, 7]
# x7 + 2x4 - 5
# 8x3 + 7x6


def poly_derivative(poly):
    """
    main
    :param poly: polynomial
    :return: derivative of polynomial
    """
    derivative = []
    for i in range(len(poly) - 1, 0, -1):
        print('poly[i]: {} i: {}'.format(poly[i], i))
        derivative.append(poly[i] * i)
    derivative.reverse()
    return derivative
