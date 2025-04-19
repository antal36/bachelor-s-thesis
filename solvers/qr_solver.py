import numpy as np
from .solver import Solver
from sklearn.linear_model import LinearRegression
from .matrix_generator import MatrixGenerator

class QR_solver(Solver):

    def __init__(self, X: np.ndarray[np.ndarray], y: np.ndarray, r: int) -> None:
        super().__init__(X, y, r)
    
    def get_leverage_scores(self) -> np.ndarray[np.float64]:
        Q, R = np.linalg.qr(self.X)

        return np.sum(np.square(Q), axis=1)
        #return np.square(np.linalg.norm(Q, axis=1))
        # return Q @ Q.T
    

if __name__ == "__main__":
    X, y, intercept, params = MatrixGenerator.lognormal(50000, 200, 0, 1)
    true_parameters: np.ndarray = np.array([intercept] + list(params))

    test = QR_solver(X=X, y=y, r=5_000)
    qr_params = test.solve(method="unweighted", method_probs="classic")
    qr_predictions = test.predict(X=X)

    lin_reg = LinearRegression()
    lin_reg.fit(X=X, y=y)
    lin_reg_params = np.array([lin_reg.intercept_] + list(lin_reg.coef_))
    lin_reg_predictions = lin_reg.predict(X=X)

    print("QR parametre MSE:", np.sum((true_parameters - qr_params)**2))
    print("lin reg parametre MSE:", np.sum((true_parameters - lin_reg_params)**2))
    print('--------------------------')

    print("QR y MSE:", np.sum((y - qr_predictions)**2))
    print("lin reg y MSE:", np.sum((y - lin_reg_predictions)**2))
    print('--------------------------')

    print("vzdialenost qr predikcii a lin reg predikcii", np.sum((lin_reg_predictions - qr_predictions)**2))





