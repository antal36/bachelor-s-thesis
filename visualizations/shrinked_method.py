import sys

sys.path.append("../")

from solvers.blendenpik_solver import BlendenpikSolver  
from solvers.qr_solver import QR_solver  
from solvers.matrix_generator import MatrixGenerator
from scripts.solvers.tester_new import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


lambdas: np.ndarray = np.arange(0.05, 1, 0.05)

mse_train: list[float] = []
mse_test: list[float] = []
mse_train_big: list[float] = []
mse_test_big: list[float] = []

results = {
    "mse_train": defaultdict(list),
    "mse_test": defaultdict(list),
    "mse_train_big": defaultdict(list),
    "mse_test_big": defaultdict(list),
}

for i in range(50):
    print(f"Iteration {i+1}/50")
    X, y, intercept, beta = MatrixGenerator.normal(rows=40_000, cols=500,
                                                   mean=0, std=1)
    X_big, y_big, intercept_big, beta_big = MatrixGenerator.normal(rows=80_000, cols=500,
                                                   mean=0, std=1)


    X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, test_size=0.5, random_seed=42)
    X_train_big, X_test_big, y_train_big, y_test_big = train_test_split(X=X_big, y=y_big, test_size=0.5, random_seed=42)

    blendenpik = BlendenpikSolver(X=X_train, y=y_train, r=1_000)
    blendenpik_big = BlendenpikSolver(X=X_train_big, y=y_train_big, r=1_000)

    lam: float
    for lam in lambdas:
        blendenpik.solve(method="unweighted", method_probs="shrinked", lam=lam)
        results["mse_train"][lam].append(np.mean((blendenpik.predict(X=X_train) - y_train) ** 2))
        results["mse_test"][lam].append(np.mean((blendenpik.predict(X=X_test) - y_test) ** 2))

        blendenpik_big.solve(method="unweighted", method_probs="shrinked", lam=lam)
        results["mse_train_big"][lam].append(np.mean((blendenpik_big.predict(X=X_train_big) - y_train_big) ** 2))
        results["mse_test_big"][lam].append(np.mean((blendenpik_big.predict(X=X_test_big) - y_test_big) ** 2))

mean_mse = {
    key: {lam: np.mean(values) for lam, values in result_dict.items()}
    for key, result_dict in results.items()
}

mse_train = [mean_mse["mse_train"][value] for value in mean_mse["mse_train"]]
mse_test = [mean_mse["mse_test"][value] for value in mean_mse["mse_test"]]
mse_train_big = [mean_mse["mse_train_big"][value] for value in mean_mse["mse_train_big"]]
mse_test_big = [mean_mse["mse_test_big"][value] for value in mean_mse["mse_test_big"]]

###############################################
# Shrinked method plot
fig, axes = plt.subplots(2, 2, sharex=True)
axes = axes.flatten()

# Plot the data
# 20_000 x 500
sns.lineplot(x=lambdas, y=mse_train, ax=axes[0])
sns.lineplot(x=lambdas, y=mse_test, ax=axes[2])

sns.lineplot(x=lambdas, y=mse_train_big, ax=axes[1])
sns.lineplot(x=lambdas, y=mse_test_big, ax=axes[3])

# Remove the individual x-labels
for ax in axes:
    ax.set_xlabel("")

axes[0].set_title(r"$20000 \times 500$")
axes[1].set_title(r"$40000 \times 500$")

# Set column titles instead of using legends
axes[0].set_ylabel(r"$MSE_{train}$")
axes[2].set_ylabel(r"$MSE_{test}$")
axes[1].set_ylabel(r"$MSE_{train}$")
axes[3].set_ylabel(r"$MSE_{test}$")
axes[2].set_xlabel(r"$\lambda$")
axes[3].set_xlabel(r"$\lambda$")

# Turn off legends
for ax in axes:
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# Adjust layout
plt.tight_layout()
# plt.subplots_adjust(top=0.9) 

plt.show()
