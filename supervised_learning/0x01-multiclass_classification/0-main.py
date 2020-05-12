#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('0-one_hot_encode').one_hot_encode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:4]

print(Y)
Y_one_hot = oh_encode(Y, 6)
print(Y_one_hot)
