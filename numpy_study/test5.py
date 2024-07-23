import numpy as np

tang_array = np.arange(10)
print(tang_array)
print(tang_array.shape)
tang_array.shape = 2, 5
print(tang_array)

tang_array.reshape(1, 10)
print(tang_array)

tang_array = np.arange(10)
print(tang_array.shape)
tang_array = tang_array[np.newaxis, :]
print(tang_array.shape)
tang_array = np.arange(10)
print(tang_array.shape)

tang_array = tang_array[:, np.newaxis]
print(tang_array.shape)

tang_array = tang_array[:, np.newaxis, np.newaxis]
print(tang_array.shape)

tang_array = tang_array.squeeze()
print(tang_array.shape)

tang_array.shape = 2, 5
print(tang_array)

print(tang_array.transpose())

print(tang_array.T)

