from .qr_solver import QR_solver
from .blendenpik_solver import BlendenpikSolver
from .iboss_solver import IBOSS_Solver
from .hadamard_solver import HadamardSolver
from .naive_solver import NaiveSolver
from sklearn.linear_model import LinearRegression
from .matrix_generator import MatrixGenerator
import time
import numpy as np
import pandas as pd
from typing import Literal, Any, Optional

np.random.seed(12)

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.5, random_seed: Optional[int] = None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_samples = int(n_samples * test_size)
    
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def solve_once(
    model_name: Literal["basic_lin_reg", "qr", "blendenpik", "blendenpik_foreign", "iboss", "hadamard", "naive"],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    r: int,
    method: Literal["weighted", "unweighted"] = "weighted",
    method_probs: Literal["classic", "shrinked", "not_specified", "None"] = "shrinked"
) -> dict[str, float]:
    """
    Run one iteration of the solver, returning MSE of train predictions,
    MSE of test predictions, and the time taken.
    """
    match model_name:
        case "basic_lin_reg":
            model = LinearRegression
        case "qr":
            model = QR_solver
        case "blendenpik":
            model = BlendenpikSolver
        case "naive":
            model = NaiveSolver
        case "iboss":
            model = IBOSS_Solver
        case "hadamard":
            model = HadamardSolver
        case _:
            raise ValueError("Incorrect solver chosen")

    start_time = time.time()

    # Solve
    if model_name == "basic_lin_reg":
        solver = model()
        solver.fit(X_train, y_train)

        train_preds = solver.predict(X_train)
        test_preds = solver.predict(X_test)

    else:
        # e.g., QR, blendenpik, iboss, hadamard, naive
        solver = model(X_train, y_train, r)
        if model_name == "iboss":
            solver.solve_2()  # IBOSS doesn't need method/method_probs
        else:
            # method & method_probs pass through
            solver.solve(method=method, method_probs=method_probs)

        train_preds = solver.predict(X_train)
        test_preds = solver.predict(X_test)

    end_time = time.time()

    # Compute MSEs
    mse_train = np.mean((y_train - train_preds) ** 2)
    mse_test = np.mean((y_test - test_preds) ** 2)

    return {
        "mse_train": mse_train,
        "mse_test": mse_test,
        "time": end_time - start_time
    }

def new_row(
    model_name: str,
    method: str,
    prob_method: str,
    mean_time: float,
    std_time: float,
    mean_mse_train: float,
    std_mse_train: float,
    mean_mse_test: float,
    std_mse_test: float
) -> dict[str, Any]:
    """Creates one row (dictionary) for the resulting DataFrame/CSV."""
    return {
        "solver": model_name,
        "method": method,
        "prob method": prob_method,
        "mean_time": mean_time,
        "std_time": std_time,
        "mean_mse_train": mean_mse_train,
        "std_mse_train": std_mse_train,
        "mean_mse_test": mean_mse_test,
        "std_mse_test": std_mse_test,
    }

def main():
    # Number of repeats per combination
    n_it: int = 50
    rows: int = 40_000
    cols: int = 500
    r: int = 1_000

    methods: dict[str, list[str]] = {
        "basic_lin_reg": ["unweighted"],
        # "naive": ["unweighted", "weighted"],
        # "qr": ["weighted", "unweighted"],
        # "blendenpik": ["weighted", "unweighted"],
        "blendenpik": ["unweighted"],
        # "hadamard": ["unweighted", "weighted"],
        "iboss": ["unweighted"]
    }
    
    probability_methods: dict[str, dict[str, list[str]]] = \
    {
        "basic_lin_reg":      {"unweighted": ["None"]},
        # "naive":                 {"weighted": ["classic", "shrinked"], "unweighted": ["classic", "shrinked"]},
        # "qr":                 {"weighted": ["classic", "shrinked"], "unweighted": ["classic", "shrinked"]},
        # "blendenpik":         {"weighted": ["classic", "shrinked"], "unweighted": ["classic", "shrinked"]},
        "blendenpik":         {"unweighted": ["shrinked"]},
        # "hadamard":           {"weighted": ["classic", "shrinked"], "unweighted": ["classic", "shrinked"]},
        "iboss":              {"unweighted": ["not_specified"]}
    }

    # -------------------------------------------------------------------------
    # 1) Build a list of solver/method/prob_method combinations we want to test
    # -------------------------------------------------------------------------
    solver_combinations: list[tuple[str]] = []
    for model_name in methods:
        for method in methods[model_name]:
            for prob_method in probability_methods[model_name][method]:
                solver_combinations.append((model_name, method, prob_method))

    # -------------------------------------------------------------------------
    # 2) Prepare a dictionary to accumulate results across multiple runs.
    #    We'll store times, MSE_train, and MSE_test for each combination.
    # -------------------------------------------------------------------------
    results_accum: dict[tuple[str], dict[str, list[str]]] = \
    {
        (model_name, method, prob_method): {
            "times": [],
            "mse_train": [],
            "mse_test": []
        }
        for (model_name, method, prob_method) in solver_combinations
    }

    # -------------------------------------------------------------------------
    # 3) Main loop over n_it (outer loop).
    #    For *each iteration*, generate new data, split into train/test,
    #    then evaluate *all* solver combinations on the same data.
    # -------------------------------------------------------------------------
    i: int
    for i in range(n_it):
        print(f"Iteration {i+1}/{n_it}")
        # Generate a new matrix for each iteration
        X, y, _, _ = MatrixGenerator.multivariate_normal_mixed(rows=rows, cols=cols)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_seed=42)

        # Evaluate each solver on this new data
        for (model_name, method, prob_method) in solver_combinations:
            print(f"Evaluating: {model_name}, {method}, {prob_method}")
            result_dict = solve_once(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                r=r,
                method=method,
                method_probs=prob_method
            )
            # Accumulate
            results_accum[(model_name, method, prob_method)]["times"].append(result_dict["time"])
            results_accum[(model_name, method, prob_method)]["mse_train"].append(result_dict["mse_train"])
            results_accum[(model_name, method, prob_method)]["mse_test"].append(result_dict["mse_test"])

    # -------------------------------------------------------------------------
    # 4) After all runs, compute mean & std for each combination, build rows
    # -------------------------------------------------------------------------
    rows_of_df: list[dict[str, float]] = []

    for (model_name, method, prob_method), data_dict in results_accum.items():
        times = data_dict["times"]
        mse_train_list = data_dict["mse_train"]
        mse_test_list = data_dict["mse_test"]

        mean_time = np.mean(times)
        std_time = np.std(times)

        mean_mse_train = np.mean(mse_train_list)
        std_mse_train = np.std(mse_train_list)

        mean_mse_test = np.mean(mse_test_list)
        std_mse_test = np.std(mse_test_list)

        row_dict: dict[str, float] = new_row(
            model_name=model_name,
            method=method,
            prob_method=prob_method,
            mean_time=mean_time,
            std_time=std_time,
            mean_mse_train=mean_mse_train,
            std_mse_train=std_mse_train,
            mean_mse_test=mean_mse_test,
            std_mse_test=std_mse_test
        )
        rows_of_df.append(row_dict)

    result_df: pd.DataFrame = pd.DataFrame(rows_of_df)
    result_df.to_csv("all_results_2703.csv", index=False)

    return 0

if __name__ == "__main__":
    main()
