#!/usr/bin/env python3
"""
derivatives for polys
"""


def poly_derivative(poly):
    derivates = []
    for i in range(len(poly) - 1, 0, -1):
        derivates.append(poly[i] * i)
    return derivates
