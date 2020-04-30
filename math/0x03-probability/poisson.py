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
        if len(data) < 2:
            self.data = data
        else:
            raise ValueError('data must contain multiple values')
        if lambtha >= 0:
            self.lambtha = float(sum(data) / len(data)) if data is not None else \
                float(lambtha)
        else:
            raise TypeError('data must be a list')
