import sys
from .solver import Solver
import numpy as np
import math
from .matrix_generator import MatrixGenerator
import time

sys.path.append(r"C:\Users\antal\Desktop\matfyz\bakalárka\scripts\fwht_implement")
import fwht #type: ignore


class HadamardSolver(Solver):

    def __init__(self, X: np.ndarray[np.ndarray], y: np.ndarray, r: int):
        super().__init__(X, y, r)
    
    def get_leverage_scores(self):
        d = np.random.choice([-1, 1], self.X.shape[0]) # vektor nahodne +- jednotiek
        d = d[:, np.newaxis] # z vektora (dimenzia, ) spravime (dimenzia, 1) teda ako by stlpcovy vektor
        matrix = d * self.X # po zlozkove nasobenie, prvou zlozkou z d vynasobi cely prvy riadok z X

        """number of columns in matrix must be power of 2"""
        rows, cols = matrix.shape
        padding = np.zeros((2 ** math.ceil(math.log2(rows)) - rows, cols))
        matrix = np.vstack((matrix, padding))

        fwht.fwht(matrix) #inplace transformation, should save same time?
        sampled_transformed = matrix[np.random.choice(matrix.shape[0], size=matrix.shape[1], replace=False)]
        """sampling the transformed matrix, sampling k rows (X \\in R_(n x k) )"""

        Q, R = np.linalg.qr(sampled_transformed)

        # Pi_2 = np.random.choice([-1, 1], size=(self.X.shape[1], int(np.log(self.X.shape[0]))))
        Pi_2 = np.random.choice([np.sqrt(3/self.r), -np.sqrt(3/self.r), 0], size=(self.X.shape[1], int(np.log(self.X.shape[0]))), p=[1/6, 1/6, 2/3])
        Omega = self.X @ (np.linalg.inv(R) @ Pi_2)

        return np.sum(np.square(Omega), axis=1)

if __name__ == "__main__":
    final_time = 0
    final_time2 = 0

    for _ in range(50):
        X, y, intercept, params = MatrixGenerator.lognormal(50000, 200, 0, 1)
        true_parameters: np.ndarray = np.array([intercept] + list(params))

        test = HadamardSolver(X=X, y=y, r=5_000)

        start_time = time.time()
        blendpik_params = test.solve(method="unweighted", method_probs="classic")
        final_time += time.time() - start_time
