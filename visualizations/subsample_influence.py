# import sys
# sys.path.append("../")

# import seaborn as sns
# from solvers.iboss_solver import IBOSS_Solver
# from solvers.blendenpik_solver import BlendenpikSolver
# from solvers.matrix_generator import MatrixGenerator
# from sklearn.linear_model import LinearRegression
# from solvers.tester_new import train_test_split
# import numpy as np
# from collections import defaultdict
# import time
# import matplotlib.pyplot as plt

# np.random.seed(42)

# list_of_r: list[int] = np.arange(1_000, 20_000, 500, dtype=int)

# results: dict[str, defaultdict[int, list[float]]] = {
#     "mse_train": defaultdict(list),
#     "mse_test": defaultdict(list),
#     "time": defaultdict(list)
# }

# n_it: int = 50
# for i in range(n_it):
#     X, y, intercept, beta = MatrixGenerator.multivariate_normal_mixed(rows=40_000, cols=500)
#     X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, test_size=0.5, random_seed=42)
#     print(f"Iteration {i+1}/{n_it}")

#     for r in list_of_r:
#         solver: BlendenpikSolver = BlendenpikSolver(X=X_train, y=y_train, r=r)

#         start_time = time.time()
#         solver.solve(method="unweighted", method_probs="shrinked", lam=0.5)
#         end_time = time.time()

#         mse_train: float = np.mean((solver.predict(X=X_train) - y_train)**2)
#         mse_test: float = np.mean((solver.predict(X=X_test) - y_test)**2)
#         duration: float = end_time - start_time

#         results["mse_train"][r].append(mse_train)
#         results["mse_test"][r].append(mse_test)
#         results["time"][r].append(duration)

# # Výpočet priemerných metrík pre podvzorkové metódy
# mean_mse = {
#     key: {lam: np.mean(values) for lam, values in result_dict.items()}
#     for key, result_dict in results.items()
# }

# mse_train: list[float] = [mean_mse["mse_train"][value] for value in mean_mse["mse_train"]]
# mse_test: list[float] = [mean_mse["mse_test"][value] for value in mean_mse["mse_test"]]
# times: list[float] = [mean_mse["time"][value] for value in mean_mse["time"]]

# ###############################################
# # Plotting

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# sns.lineplot(x=list_of_r, y=mse_train, ax=axes[0], marker="o", color="skyblue")
# axes[0].axhline(0.010, linestyle="--", color="red", label="Linear Regression")
# axes[0].set_title(r"$MSE_{train}$", fontsize=14)
# axes[0].legend()

# sns.lineplot(x=list_of_r, y=mse_test, ax=axes[1], marker="o", color="darkgreen")
# axes[1].axhline(0.010, linestyle="--", color="red", label="Linear Regression")
# axes[1].set_title(r"$MSE_{test}$", fontsize=14)
# axes[1].legend()

# sns.lineplot(x=list_of_r, y=times, ax=axes[2], marker="o", color="orangered")
# axes[2].axhline(0.427, linestyle="--", color="red", label="Linear Regression")
# axes[2].set_title("Čas [s]", fontsize=14)
# axes[2].legend()

# fig.text(0.50, 0.02, r'Veľkosť podvzorky', ha='center', fontsize=14)
# plt.tight_layout()
# plt.show()



import sys
sys.path.append("../")

import seaborn as sns
from solvers.iboss_solver import IBOSS_Solver
from solvers.blendenpik_solver import BlendenpikSolver
from solvers.matrix_generator import MatrixGenerator
from solvers.tester_new import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt

np.random.seed(42)

# Subsample sizes to sweep
list_of_r = np.arange(1_000, 20_000, 500, dtype=int)

# Prepare results storage per solver
results = {
    "Blendenpik": {"mse_train": defaultdict(list), "mse_test": defaultdict(list), "time": defaultdict(list)},
    "IBOSS":      {"mse_train": defaultdict(list), "mse_test": defaultdict(list), "time": defaultdict(list)},
}

# For OLS baseline
lr_mse_train = []
lr_mse_test  = []
lr_times     = []

n_it = 50
for i in range(n_it):
    X, y, intercept, beta = MatrixGenerator.multivariate_normal_mixed(rows=40_000, cols=500)
    X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, test_size=0.5, random_seed=42)
    print(f"Iteration {i+1}/{n_it}")

    # --- Fit full OLS baseline once per iteration ---
    lr = LinearRegression()
    t0_lr = time.time()
    lr.fit(X_train, y_train)
    t1_lr = time.time()

    lr_pred_tr = lr.predict(X_train)
    lr_pred_te = lr.predict(X_test)
    lr_mse_train.append(np.mean((lr_pred_tr - y_train)**2))
    lr_mse_test.append( np.mean((lr_pred_te - y_test)**2))
    lr_times.append(t1_lr - t0_lr)

    for r in list_of_r:
        # --- Blendenpik ---
        bp = BlendenpikSolver(X=X_train, y=y_train, r=r)
        t0 = time.time()
        bp.solve(method="unweighted", method_probs="shrinked", lam=0.5)
        t1 = time.time()

        preds_train_bp = bp.predict(X_train)
        preds_test_bp  = bp.predict(X_test)

        results["Blendenpik"]["time"][r].append(t1 - t0)
        results["Blendenpik"]["mse_train"][r].append(np.mean((preds_train_bp - y_train)**2))
        results["Blendenpik"]["mse_test"][r].append(np.mean((preds_test_bp  - y_test )**2))

        # --- IBOSS ---
        ib = IBOSS_Solver(X=X_train, y=y_train, r=r)
        t0 = time.time()
        ib.solve()   # add args if needed
        t1 = time.time()

        preds_train_ib = ib.predict(X_train)
        preds_test_ib  = ib.predict(X_test)

        results["IBOSS"]["time"][r].append(t1 - t0)
        results["IBOSS"]["mse_train"][r].append(np.mean((preds_train_ib - y_train)**2))
        results["IBOSS"]["mse_test"][r].append(np.mean((preds_test_ib  - y_test )**2))

# Compute per-r averages for sub‐sampling methods
mean_stats = {
    solver: {
        metric: {r: np.mean(vals) for r, vals in data[metric].items()}
        for metric in ("mse_train", "mse_test", "time")
    }
    for solver, data in results.items()
}

# Compute average OLS baseline
mean_lr_train = np.mean(lr_mse_train)
mean_lr_test  = np.mean(lr_mse_test)
mean_lr_time  = np.mean(lr_times)

# Build aligned series
bp_train = [mean_stats["Blendenpik"]["mse_train"][r] for r in list_of_r]
bp_test  = [mean_stats["Blendenpik"]["mse_test"][r]  for r in list_of_r]
bp_time  = [mean_stats["Blendenpik"]["time"][r]      for r in list_of_r]

ib_train = [mean_stats["IBOSS"]["mse_train"][r] for r in list_of_r]
ib_test  = [mean_stats["IBOSS"]["mse_test"][r]  for r in list_of_r]
ib_time  = [mean_stats["IBOSS"]["time"][r]      for r in list_of_r]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Train MSE
sns.lineplot(x=list_of_r, y=bp_train, ax=axes[0], marker="o", label="Zrazená vplyvová metóda")
sns.lineplot(x=list_of_r, y=ib_train, ax=axes[0], marker="s", label="Metóda IBOSS")
axes[0].axhline(mean_lr_train, linestyle="--", color="red", label="Úplná regresia")
axes[0].set_title(r"$MSE_{train}$", fontsize=14)
axes[0].set_xlabel("Veľkosť podvzorky")
axes[0].legend()

# Test MSE
sns.lineplot(x=list_of_r, y=bp_test, ax=axes[1], marker="o", label="Zrazená vplyvová metóda")
sns.lineplot(x=list_of_r, y=ib_test, ax=axes[1], marker="s", label="Metóda IBOSS")
axes[1].axhline(mean_lr_test, linestyle="--", color="red", label="Úplná regresia")
axes[1].set_title(r"$MSE_{test}$", fontsize=14)
axes[1].set_xlabel("Veľkosť podvzorky")
axes[1].legend()

# Time
sns.lineplot(x=list_of_r, y=bp_time, ax=axes[2], marker="o", label="Zrazená vplyvová metóda")
sns.lineplot(x=list_of_r, y=ib_time, ax=axes[2], marker="s", label="Metóda IBOSS")
axes[2].axhline(mean_lr_time, linestyle="--", color="red", label="Úplná regresia")
axes[2].set_title("Čas [s]", fontsize=14)
axes[2].set_xlabel("Veľkosť podvzorky")
axes[2].legend()

plt.tight_layout()
plt.show()
