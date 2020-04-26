#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [-5, 0, 0, 0, 2, 0, 0, 1]
# poly = [5, 3, 0, 1]
# 8x3 + 7x6
print(poly_derivative(poly))
