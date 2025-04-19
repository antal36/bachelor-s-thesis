from sympy import fwht
import transformation # type: ignore
import transformation_strided # type: ignore
import fwht # type: ignore
import numpy as np
import math
import time


np.random.seed(123)


# start_time = time.time()
# for i in range(1):
        
#     matrix = np.random.normal(0, 1, (5000, 200))
#     transformed = np.apply_along_axis(fwht, 0, matrix)

# print("cas sympy:", time.time() - start_time)

# #------------

start_time = time.time()

for j in range(1):
    matrix = np.random.normal(0, 1, (5, 2))
    rows, cols = matrix.shape
    padding = np.zeros((2 ** math.ceil(math.log2(rows)) - rows, cols))
    matrix = np.vstack((matrix, padding))
    transformed_one = transformation.fwht(matrix)
    fwht.fwht(matrix)
    print(matrix)

    print(np.allclose(transformed_one, matrix))

# print(np.apply_along_axis(fwht, 0, matrix))