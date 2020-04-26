#!/usr/bin/env python3
"""
sigma notation
"""


def _summation_i_squared(n):
    """
    recursive helper
    :param n: limit to sum
    :return: nothing
    """
    global counter
    global result
    result = result + n ** 2
    if counter == limit:
        return
    counter = counter + 1
    _summation_i_squared(counter)


def summation_i_squared(n):
    """
    main
    :param n: limit to sum
    :return: sum
    """
    global limit
    global counter
    global result
    limit = n
    counter = 1
    result = 0

    if n < 1 or n is None:
        return None
    _summation_i_squared(1)
    return result