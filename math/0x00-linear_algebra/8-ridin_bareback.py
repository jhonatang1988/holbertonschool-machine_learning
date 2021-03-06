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
    """
    main
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    twodimadded = []
    flatList = [i + j for i, j in zip(mat1, mat2)]
    for i in flatList:
        onedimadded = []
        for j in range(len(mat1[0])):
            onedimadded.append(sum(i[j::len(mat1[0])]))
        twodimadded.append(onedimadded)
    return twodimadded


"""
concatenates two arrays
"""


def cat_arrays(arr1, arr2):
    """
    main
    """
    return arr1 + arr2


"""
concatenates two matrices based on axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    main
    """
    newMat1 = []
    newMat2 = []
    for i in mat1:
        newMat1.append(i[:])
    for i in mat2:
        newMat2.append(i[:])
    if axis == 0:
        if len(newMat1[0]) != len(newMat2[0]):
            return None
        newList = newMat1 + newMat2
        return newList
    if axis == 1:
        if len(newMat1) != len(newMat2):
            return None
        flatList = [i + j for i, j in zip(newMat1, newMat2)]
        return flatList


"""
matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    main
    """
    if len(mat1[0]) != len(mat2):
        return None
    preList = []
    for column in range(len(mat2[0])):
        outerList = []
        for row in range(len(mat1)):
            innerList = []
            for k in range(len(mat1[0])):
                innerList.append(mat1[row][k] * mat2[k][column])
            outerList.append(sum(innerList))
        preList.append(outerList)
    finalList = []
    for i in range(len(preList[0])):
        finalList.append([el[i] for el in preList])
    return finalList
