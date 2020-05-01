#!/usr/bin/env python3
"""
represents a binomial distribution
"""


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        if n <= 0:
            raise ValueError('n must be a positive value')
        elif 0 >= p <= 1:
            raise ValueError('p must be greater than 0 and less than 1')
        elif data is None:
            self.n = n
            self.p = p
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
                self.n = int(round(self.mean / self.p))
                self.p = (self.mean / self.n)
            except TypeError:
                print('could not convert to int')
