#!/usr/bin/env python3
"""
get the integrate
"""


def poly_integral(poly, C=0):
    """
    :param poly: polynomial
    :param C: constant
    :return: integrate
    """
    if type(poly) is not list or len(poly) == 0 or (type(C) is not int and
                                                    type(C) is not float):
        return None

    for num in poly:
        if type(num) is not int and type(num) is not float:
            return None

    if poly == [0]:
        return [C]

    preintegrate = []
    integrate = []

    for i in range(len(poly)):
        try:
            preintegrate.append(poly[i] / (i + 1))
        except ZeroDivisionError:
            preintegrate.append(0)
    for i in preintegrate:
        integrate.append(formatnumber(i))
    integrate.insert(0, C)
    return integrate


"""
strip .0 when necessary
"""


def formatnumber(num):
    """
    :param num: float
    :return: float without .0 when necessary
    """
    if num % 1 == 0:
        return int(num)
    else:
        return num
