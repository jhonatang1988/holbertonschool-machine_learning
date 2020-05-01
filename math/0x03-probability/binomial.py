#!/usr/bin/env python3
"""
represents a binomial distribution
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


class Binomial:
    """
    class Binomial
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        :param data: list of the data to be used to estimate the distribution
        :param n: number of Bernoulli trials
        :param p: probability of a “success”
        """
        if n <= 0:
            raise ValueError('n must be a positive value')
        elif 0 > p < 1:
            raise ValueError('p must be greater than 0 and less than 1')
        elif data is None:
            self.n = toint(n)
            self.p = toFloat(p)
        elif type(data) is not list:
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            self.mean = sum(data) / len(data)
            variance = 0
            for i in data:
                variance += (i - self.mean) ** 2
            variance = variance / len(data)
            self.p = 1 - (variance / self.mean)
            try:
                self.n = toint(round(self.mean / self.p))
                self.p = toFloat(self.mean / self.n)
            except TypeError:
                print('could not convert to int')
