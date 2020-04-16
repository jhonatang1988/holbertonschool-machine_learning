#!/usr/bin/env python3
"""
wrapper
"""


def _matrix_shape(matrix):
    """
    wrapper for keep shape state
    """
    global shape
    for i in matrix:
        if type(i) is list:
            shape.append(len(i))
            _matrix_shape(i)
        return shape


"""
main
"""


def matrix_shape(matrix):
    """
    main
    """
    global shape
    shape = [len(matrix)]
    return _matrix_shape(matrix)
