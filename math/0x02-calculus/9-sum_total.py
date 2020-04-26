#!/usr/bin/env python3
def _summation_i_squared(n):
    global counter
    global result
    result = result + n**2
    if counter == limit:
        return
    counter = counter + 1
    _summation_i_squared(counter)

def summation_i_squared(n):
    global limit
    global counter
    global result
    limit = n
    counter = 1
    result = 0

    if n < 1:
        return None
    _summation_i_squared(1)
    return result
