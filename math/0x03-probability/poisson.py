#!/usr/bin/env python3
"""
represents a poisson distribution
"""


class Poisson:
    """
    poisson distribution class
    """

    def __init__(self, data=None, lambtha=1.):
        self.data = data
        self.lambtha = sum(data) / len(data) if data is not None else \
            float(lambtha)
