#!/usr/bin/env python3

add_matrices2D = __import__('5-across_the_planes').add_matrices2D

mat1 = [[1.9, 2.0, 3.5], [3.9, 4.5, 4.3]]
mat2 = [[5.5, 6.4, 5.4], [7.3, 8.5, 6.5]]
print(add_matrices2D(mat1, mat2))
print(mat1)
print(mat2)
print(add_matrices2D(mat1, [[1, 2], [4, 5]]))
