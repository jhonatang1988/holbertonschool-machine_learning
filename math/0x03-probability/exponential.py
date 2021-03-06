#!/usr/bin/env python3
"""
class Exponential that represents an exponential distribution
"""


def toFloat(lambtha):
    """
    :param lambtha: to convert to float
    :return: float or exception
    """
    try:
        return float(lambtha)
    except TypeError:
        print('it must be convertible to float')


class Exponential:
    """
    Exponential class
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        :param data: is a list of the data to be used to estimate the
        distribution :param lambtha: is the expected number of occurences in
        a given time frame
        """
        if lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        elif data is None:
            self.lambtha = toFloat(lambtha)
        elif type(data) is not list:
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        :param x: time period
        :return: the probability density function
        """
        if x < 0:
            return 0
        return (Exponential.e ** (-self.lambtha * x)) * self.lambtha

    def cdf(self, x):
        """
        :param x: time period
        :return: cumulative distribution function
        """
        if x < 0:
            return 0
        return 1 - (Exponential.e ** (-self.lambtha * x))
