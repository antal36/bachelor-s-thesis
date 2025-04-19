import numpy as np
from .solver import Solver
from .matrix_generator import MatrixGenerator
from sklearn.linear_model import LinearRegression
import time
import math



class BlendenpikSolver(Solver):

    def __init__(self, X: np.ndarray[np.ndarray], y: np.ndarray, r: int) -> None:
        super().__init__(X, y, r)
    
    def get_leverage_scores(self) -> np.ndarray:
        #NOTE we set r1 = p; 1:5p; 2p; 3p; 5p
        #NOTE r2 = \\kappa log(n), for \\kappa = 1; 2; 3; 4; 5; 10; 20,
        #NOTE These observations suggest that we may use a combination of small r1 and large r2 to achieve high-quality approximation and short running time.
        
        Pi_1 = np.random.choice([-1, 1], size=(self.X.shape[1], self.X.shape[0]))

        Q1, R1 = np.linalg.qr(Pi_1 @ self.X)

        Pi_2 = np.random.choice([np.sqrt(3/self.r), -np.sqrt(3/self.r), 0], size=(self.X.shape[1], int(np.log(self.X.shape[0]))), p=[1/6, 1/6, 2/3])

        Omega = self.X @ (np.linalg.inv(R1) @ Pi_2)
        
        return np.sum(np.square(Omega), axis=1)
        # return np.square(np.linalg.norm(Omega, axis=1))
        # return Omega @ Omega.T


if __name__ == "__main__":
    final_time = 0
    final_time2 = 0

    for _ in range(50):
        X, y, intercept, params = MatrixGenerator.lognormal(50000, 200, 0, 1)
        true_parameters: np.ndarray = np.array([intercept] + list(params))

        test = BlendenpikSolver(X=X, y=y, r=5_000)

        start_time = time.time()
        blendpik_params = test.solve(method="unweighted", method_probs="classic")
        final_time += time.time() - start_time

    #     blendpik_predictions = test.predict(X=X)

    #     lin_reg = LinearRegression()

    #     start_time = time.time()
    #     lin_reg.fit(X=X, y=y)
    #     final_time2 += time.time() - start_time

    #     lin_reg_params = np.array([lin_reg.intercept_] + list(lin_reg.coef_))
    #     lin_reg_predictions = lin_reg.predict(X=X)

    # print("cas blendenpik:", final_time)
    # print("cas lin_reg:", final_time2)
    # print('--------------------------')


    # print("QR parametre MSE:", np.sum((true_parameters - blendpik_params)**2))
    # print("lin reg parametre MSE:", np.sum((true_parameters - lin_reg_params)**2))
    # print('--------------------------')

    # print("QR y MSE:", np.sum((y - blendpik_predictions)**2))
    # print("lin reg y MSE:", np.sum((y - lin_reg_predictions)**2))
    # print('--------------------------')

    # print("vzdialenost qr predikcii a lin reg predikcii", np.sum((lin_reg_predictions - blendpik_predictions)**2))
