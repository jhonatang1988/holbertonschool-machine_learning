#!/usr/bin/env python3
"""
transpose manual
"""


def matrix_transpose(matrix):
    """
    get the transpose without libraries
    """
    i = 0
    transpose = []
    while i < len(matrix[0]):
        transpose.append([row[i] for row in matrix])
        i = i + 1
    return transpose
