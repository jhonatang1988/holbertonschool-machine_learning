#!/usr/bin/env python3
"""
represents a normal distribution
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


class Normal:
    """
    normal distribution
    """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        :param data: list of the data to be used to estimate the distribution
        :param mean: mean of the distribution
        :param stddev: standard deviation of the distribution
        """
        if stddev <= 0:
            raise ValueError('stddev must be a positive value')
        elif data is None:
            self.mean = toFloat(mean)
            self.stddev = toFloat(stddev)
        elif type(data) is not list:
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            total = 0.0
            self.mean = sum(data) / len(data)
            for i in data:
                total += (float(i) - self.mean) ** 2
            self.stddev = (total / len(data)) ** (1 / 2)

    def z_score(self, x):
        """
        :param x: x-value
        :return: z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        :param z: the z-score
        :return: the value of z_score
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        :param x: x-value
        :return: PDF value for x
        """
        return (1 / (self.stddev * ((2 * Normal.pi) ** (1 / 2)))) * \
               (Normal.e ** (-((x - self.mean) ** 2) / (2 * (self.stddev **
                                                             2))))
