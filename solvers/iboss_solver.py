import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional
from .matrix_generator import MatrixGenerator

class IBOSS_Solver:
    X: np.ndarray[np.ndarray]
    Y: np.ndarray
    r: int
    q: int
    indices_: np.ndarray[bool]
    intercept_: Optional[np.float64]
    coef_: Optional[np.ndarray[np.float64]]


    def __init__(self, X: np.ndarray[np.ndarray], y: np.ndarray, r: int) -> None:
        self.X = X
        self.y = y

        assert 0 < r < X.shape[0], "Size of subsample must be in interval (0, # of rows)"
        assert isinstance(self.X, np.ndarray), "Feature matrix X must be a numpy array"
        assert isinstance(self.X, np.ndarray), "Target y must be a numpy array"

        self.r = r
        self.q = max(1, int(self.r / (2 * self.X.shape[1])))

        self.indices_ = np.ones(X.shape[0], dtype=bool)
        self.intercept_ = None
        self.coef_ = None

    def solve(self) -> list[np.float64]:
        for col in range(self.X.shape[1]):
            column = self.X[:, col]
            available_indices = np.nonzero(self.indices_)[0]
            available_values = column[self.indices_]

            min_qth_value = np.partition(available_values, self.q - 1)[self.q - 1]
            min_mask = available_values <= min_qth_value
            min_indices = available_indices[min_mask]

            max_q = len(available_values) - self.q
            max_qth_value = np.partition(available_values, max_q)[max_q]
            max_mask = available_values >= max_qth_value
            max_indices = available_indices[max_mask]

            # Update indices
            self.indices_[min_indices] = False
            self.indices_[max_indices] = False

        X = self.X[~self.indices_, :]
        y = self.y[~self.indices_]

        lin_reg = LinearRegression()
        lin_reg.fit(X, y)

        self.intercept_ = lin_reg.intercept_
        self.coef_ = lin_reg.coef_.copy()

        return [self.intercept_] + list(self.coef_)

    def solve_2(self) -> list[np.float64]:
        for col in range(self.X.shape[1]):
            column = self.X[:, col]

            # Indices of the rows that haven't been removed yet
            available_indices = np.nonzero(self.indices_)[0]
            available_values = column[self.indices_]

            # -- Pick the q smallest --
            # argpartition up to q-1 gives us the q smallest entries
            # but we slice [:self.q] to make sure we only take q indices
            if len(available_values) > self.q:
                min_arg_part = np.argpartition(available_values, self.q - 1)[: self.q]
            else:
                # If fewer than q remain, just take them all
                min_arg_part = np.arange(len(available_values))

            # Convert local indices (within available_values) to global row indices
            min_indices = available_indices[min_arg_part]

            # -- Pick the q largest --
            # Similarly, we partition from the end
            if len(available_values) > self.q:
                max_arg_part = np.argpartition(available_values, -self.q)[-self.q:]
            else:
                # If fewer than q remain, just take them all
                max_arg_part = np.arange(len(available_values))

            max_indices = available_indices[max_arg_part]

            # Mark those chosen rows (lowest + highest) as False in self.indices_
            self.indices_[min_indices] = False
            self.indices_[max_indices] = False

        # Build final subsample (the removed rows)
        X_subsample = self.X[~self.indices_, :]
        y_subsample = self.y[~self.indices_]

        # Fit linear regression
        lin_reg = LinearRegression()
        lin_reg.fit(X_subsample, y_subsample)

        self.intercept_ = lin_reg.intercept_
        self.coef_ = lin_reg.coef_.copy()

        # Return [intercept, coef1, coef2, ...]
        return [self.intercept_] + list(self.coef_)
    
    def predict(self, X: np.ndarray[np.ndarray]):
        if self.intercept_ is None or self.coef_ is None:
            raise ValueError("Trying to predict on unfit model")
        
        return self.intercept_ + X @ self.coef_


if __name__ == "__main__":
    X, y, intercept, params = MatrixGenerator.normal(50000, 200, 0, 1)
    true_parameters: np.ndarray = np.array([intercept] + list(params))

    test = IBOSS_Solver(X=X, y=y, r=5_000)
    iboss_params = test.solve()
    iboss_predictions = test.predict(X=X)

    lin_reg = LinearRegression()
    lin_reg.fit(X=X, y=y)
    lin_reg_params = np.array([lin_reg.intercept_] + list(lin_reg.coef_))
    lin_reg_predictions = lin_reg.predict(X=X)

    print("IBOSS parametre MSE:", np.sum((true_parameters - iboss_params)**2))
    print("lin reg parametre MSE:", np.sum((true_parameters - lin_reg_params)**2))
    print('--------------------------')

    print("IBOSS y MSE:", np.sum((y - iboss_predictions)**2))
    print("lin reg y MSE:", np.sum((y - lin_reg_predictions)**2))
    print('--------------------------')

    print("vzdialenost IBOSS predikcii a lin reg predikcii", np.sum((lin_reg_predictions - iboss_predictions)**2))






