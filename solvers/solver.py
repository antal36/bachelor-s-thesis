import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Literal, Optional

class Solver:
    X: np.ndarray[np.ndarray[np.float64]]
    y: np.ndarray[np.float64]
    intercept_: Optional[np.float64]
    coef_: Optional[np.ndarray[np.float64]]
    indices_: Optional[np.ndarray[np.float64]]

    def __init__(self, X: np.ndarray[np.ndarray], y: np.ndarray, r: int) -> None:
        """Parameters:
                    X: feature matrix 
                    y: outcome vector
                    r: size of subsample"""
        
        assert 0 < r < X.shape[0], "Size of subsample must be in interval (0, # of rows)"

        self.r = r
        self.X = X
        self.y = y

        self.intercept_ = None
        self.coef_ = None
        self.indices_ = None
    
    def get_leverage_scores(self) -> np.ndarray[np.float64]:
        """Method to obtain either exact or approx. leverage scores.
           To be implemented in subclasses"""
        
        raise NotImplementedError
    
    def get_probs(self) -> np.ndarray:
        """Method to obtain probability for leverage method"""

        leverage_scores = self.get_leverage_scores()

        return leverage_scores / np.sum(leverage_scores)
    
    def get_probs_shrinked(self, lam: float) -> np.ndarray:
        assert 0 < lam < 1, "Parameter lambda must be from interval (0, 1)"

        leveraging_scores = self.get_leverage_scores()

        return (lam * leveraging_scores / np.sum(leveraging_scores)) + ((1 - lam) / len(self.X))
    
    def get_probs_uniform(self) -> np.ndarray:

        return np.repeat(1/len(self.X), len(self.X))
    
    def solve(self, method: Literal["weighted", "unweighted"],
              method_probs: Literal["classic", "shrinked", "uniform"], **kwargs) -> list[np.float64]:
        
        lam: float = kwargs.get("lam", 0.5)

        probs_pi: np.ndarray
        match method_probs:
            case "classic":
                probs_pi = self.get_probs()
            case "shrinked":
                probs_pi = self.get_probs_shrinked(lam=lam)
            case "uniform":
                probs_pi = self.get_probs_uniform()
            case _:
                raise ValueError("Choose correct method for probability calculation")
            
        sampled_indexes: list[int] = np.random.choice(range(0, len(self.X)), size=self.r, p=probs_pi, replace=True)
        """sampling indices to make sampled matrix X and sampled vector y, also to retrieve probability \\pi_i for matrix \\Phi"""

        X_sampled = self.X[sampled_indexes, :]
        y_sampled = self.y[sampled_indexes]
        weights = probs_pi[sampled_indexes]

        self.indices_ = sampled_indexes
        
        if method == "weighted":
            linear_model = LinearRegression()
            linear_model.fit(X=X_sampled, y=y_sampled, sample_weight=weights)

            self.intercept_ = linear_model.intercept_
            self.coef_ = linear_model.coef_.copy()

            return [linear_model.intercept_] + list(linear_model.coef_)
        
        elif method == "unweighted":
            linear_model = LinearRegression()
            linear_model.fit(X=X_sampled, y=y_sampled)

            self.intercept_ = linear_model.intercept_
            self.coef_ = linear_model.coef_.copy()

            return [linear_model.intercept_] + list(linear_model.coef_)
        
        else:
            raise ValueError("Choose correct method for least square calculation!")
    
    def predict(self, X: np.ndarray[np.ndarray]) -> np.ndarray[np.float64]:
        if self.intercept_ is None or self.coef_ is None:
            raise ValueError("Trying to predict on unfit model")
        
        return self.intercept_ + X @ self.coef_
