#!/usr/bin/env python3
"""
create placeholders
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    create placeholders
    nx: param
    classes: param
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y
