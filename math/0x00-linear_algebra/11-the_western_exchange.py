#!/usr/bin/env python3
"""
returns transpose and no view
"""


def np_transpose(matrix):
    """
    main
    """
    newNdArray = matrix.copy()
    return newNdArray.T
