#!/usr/bin/env python3
"""
derivatives for polys
[5, 3, 0, 1]
"""


def poly_derivative(poly):
    """
    main
    :param poly: polynomial
    :return: derivative of polynomial
    """
    derivative = []
    for i in range(len(poly) - 1, 0, -1):
        print(i)
        derivative.append(poly[i] * i)
    derivative.reverse()
    return derivative
