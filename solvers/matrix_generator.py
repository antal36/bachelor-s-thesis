import numpy as np

class MatrixGenerator:
    beta_0: np.ndarray
    X: np.ndarray[np.ndarray[np.float64]]
    y: np.ndarray
    intercept: float

    @staticmethod
    def normal(rows: int, cols: int, mean: float, std: float):
        beta = np.random.normal(0, 1, cols)
        intercept = np.random.normal(0, 1)
        X = np.random.normal(mean, std, (rows, cols))
        y = X @ beta + intercept + np.random.normal(0, 0.1, rows)
        return X, y, intercept, beta
    
    @staticmethod
    def exponential(rows: int, cols: int, lam: float):
        beta = np.random.normal(0, 1, cols)
        intercept = np.random.normal(0, 1)
        X = np.random.exponential(lam, (rows, cols))
        y = X @ beta + intercept + np.random.normal(0, 0.1, rows)
        return X, y, intercept, beta
    
    @staticmethod
    def lognormal(rows: int, cols: int, mean: float, std: float):
        beta = np.random.normal(0, 1, cols)
        intercept = np.random.normal(0, 1)
        X = np.random.lognormal(mean, std, (rows, cols))
        y = X @ beta + intercept + np.random.normal(0, 0.1, rows)
        return X, y, intercept, beta
    
    @staticmethod
    def multivariate_normal(rows: int, cols: int, mean: np.ndarray[float], cov: np.ndarray[np.ndarray]):
        beta = np.random.normal(0, 1, cols)
        intercept = np.random.normal(0, 1)
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=(rows, cols))
        y = X @ beta + intercept + np.random.normal(0, 0.1, rows)
        return X, y, intercept, beta
    
    @staticmethod
    def poisson(rows: int, cols: int, lam: float):
        beta = np.random.normal(0, 1, cols)
        intercept = np.random.normal(0, 1)
        X = np.random.poisson(lam, (rows, cols))
        y = X @ beta + intercept + np.random.normal(0, 0.1, rows)
        return X, y, intercept, beta
    
    @staticmethod
    def multivariate_normal_mixed(rows: int, cols: int):
        cov = np.ones((cols, cols)) - np.repeat(0.5, cols**2).reshape(cols, cols) + np.diag(np.repeat(0.5, cols))
        # cov = np.eye(cols)

        # def is_pos_def(x):
        #     return np.all(np.linalg.eigvals(x) >= 0)
        
        # print(is_pos_def(cov))
        

        submatrices: list[np.ndarray] = []

        for _ in range(5):
            submatrices.append(np.random.multivariate_normal(mean=np.random.uniform(-5, 5, size=cols), cov=cov, size=(rows // 5)))

        X = np.concatenate(submatrices)

        beta = np.random.normal(0, 1, cols)
        intercept = np.random.normal(0, 1)
        y = X @ beta + intercept + np.random.normal(0, 0.1, rows)

        return X, y, intercept, beta

if __name__ == "__main__":
    X, y, intercept, beta = MatrixGenerator.multivariate_normal_mixed(rows=500, cols=2)

    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()