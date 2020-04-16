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


"""
adds two array element-wise
"""


def add_arrays(arr1, arr2):
    """
    main
    """
    if matrix_shape(arr1) != matrix_shape(arr2):
        return None
    return [(i + j) for i, j in zip(arr1, arr2)]


"""
adds two 2d matrices
"""


def add_matrices2D(mat1, mat2):
    twodimadded = []
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    flatList = [i + j for i, j in zip(mat1, mat2)]
    alist = [sum(flatList[0][::2]), sum(flatList[0][1::2])]
    blist = [sum(flatList[1][::2]), sum(flatList[1][1::2])]
    twodimadded.append(alist)
    twodimadded.append(blist)
    return twodimadded
