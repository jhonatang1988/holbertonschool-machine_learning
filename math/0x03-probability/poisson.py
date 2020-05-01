#!/usr/bin/env python3
"""
represents a poisson distribution
"""


class Poisson:
    """
    poisson distribution class
    """

    def __init__(self, data=None, lambtha=1.):
        """
        :param data: data
        :param lambtha: expected number of ocurrences
        """
        if data is None:
            self.lambtha = float(lambtha)
        elif type(data) is not list:
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        elif lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        else:
            self.lambtha = float(sum(data) / len(data))
