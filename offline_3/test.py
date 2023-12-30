import numpy as np

# test
x = np.array([[1, 2, 3],
             [4, 5, 6]])

print(x.shape)
print(x.max(axis=0))
print(x.max(axis=1))

print (x - x.max(axis=0))