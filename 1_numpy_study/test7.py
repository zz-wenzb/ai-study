import numpy as np

a = np.arange(6, dtype=int)
b = a.reshape(3, 2)
print(b)

c = np.arange(2)
print(c)

d = b.dot(c)
print(d)
