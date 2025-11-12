import numpy as np


liste = ['foo']

n = 1

while n < 10000:
    if n % 2 == 0:
        liste.append('bar')
    else:
        liste.append('foo')
    n += 1

array = np.array([liste])

print(liste, "\n", array)
