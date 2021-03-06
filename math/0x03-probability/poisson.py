#!/usr/bin/env python3
"""
represents a poisson distribution
"""


def toint(k):
    """
    :param k: param to convert to int
    :return: int or exception
    """
    try:
        k = int(k)
        return k
    except TypeError:
        print('value must be an integer or float')


class Poisson:
    """
    poisson distribution class
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        :param data: data
        :param lambtha: expected number of ocurrences
        """
        if lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        elif data is None:
            self.lambtha = float(lambtha)
        elif type(data) is not list:
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        probability mass function
        :param k: number of successes
        :return: pmf
        """
        if k < 0:
            return 0
        k = toint(k)
        fact = 1
        for i in range(1, k + 1):
            fact = fact * i
        total = Poisson.e ** (-self.lambtha) * (self.lambtha ** k) / fact
        return total

    def cdf(self, k):
        """
        :param k: number of “successes”
        :return: cumulative distribution function
        """
        if k < 0:
            return 0
        k = toint(k)
        total = 0
        for i in range(0, k + 1):
            total = total + self.pmf(i)
        return total
