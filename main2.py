
import numpy as np

import time
from _LIB_Core import timeit

a = list(range(10000))
b = list(range(100))


@timeit(1000000)
def test():
    global a
    a = a[100:]
    a += b

test()