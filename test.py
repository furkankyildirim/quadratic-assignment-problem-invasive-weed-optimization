import numpy as np

file = open('data/8-wil100.txt', 'r')

size = int(file.readline())
print(f'Sample size: {size}')
file.readline()

# Read the Flow matrix
F = np.matrix([[int(i) for i in file.readline().split()] for _ in range(size)], dtype=object)
file.readline()

# Read the Distance matrix
D = np.matrix([[int(i) for i in file.readline().split()] for _ in range(size)], dtype=object)
file.close()

p = [10, 31, 20, 59, 0, 38, 56, 63, 47, 94, 73, 49, 29, 33, 88, 26, 3, 77, 70, 18, 8, 78, 44, 40, 98, 62, 32, 6, 55, 24, 97, 51, 4, 95, 52, 76, 13, 53, 15, 48, 66, 72, 64, 75, 30, 37, 89, 87, 46, 60, 39, 99, 22, 81, 9, 92, 82, 45, 5, 50, 84, 67, 17, 41, 7, 96, 1, 69, 90, 36, 80, 21, 16, 61, 71, 57, 42, 54, 35, 19, 83, 43, 14, 34, 27, 2, 74, 28, 79, 93, 86, 25, 85, 12, 23, 68, 58, 65, 11, 91]

cost = sum([F[i, j] * D[p[i], p[j]] for i in range(size) for j in range(size)])
print(cost)