import numpy as np
from .solver import Solver
from sklearn.linear_model import LinearRegression


class NaiveSolver(Solver):

    def __init__(self, X: np.ndarray, y: np.ndarray, r: int) -> None:
        super().__init__(X, y, r)
    
    def get_leverage_scores(self) -> np.ndarray[np.float64]:

        omega = self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T

        return np.diag(omega)
