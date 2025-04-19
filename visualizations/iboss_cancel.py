import numpy as np
from solvers.iboss_solver import IBOSS_Solver
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from solvers.blendenpik_solver import BlendenpikSolver

# sns.set_theme(style='whitegrid')

np.random.seed(42)

# ------------------------------------------------
# Generate Data
# ------------------------------------------------
x_1 = np.random.uniform(0, 1, 500)
x_2 = np.where((x_1 < 0.3) | (x_1 > 0.7), 
               x_1 + np.random.normal(0, 0.02, len(x_1)), 
               x_1)

max_x_2_first_segment = np.max(x_2[x_1 < 0.3])
min_x_2_last_segment  = np.min(x_2[x_1 > 0.7])

# For the middle segment, replace x_2 values with uniform noise in [max_x_2_first_segment, min_x_2_last_segment]
in_middle = (x_1 >= 0.3) & (x_1 <= 0.7)
x_2[in_middle] = np.random.uniform(max_x_2_first_segment, 
                                   min_x_2_last_segment, 
                                   size=np.sum(in_middle))

X = np.column_stack((x_1, x_2))

# True beta and y
beta = np.random.normal(0, 1, 2)
y = X[:, 0] * beta[0] + X[:, 1] * beta[1] + np.random.normal(0, 0.5, len(X[:, 0]))

print("Real beta coefs: ", beta)

# ------------------------------------------------
# Full-data regression
# ------------------------------------------------
print("Coeffs: ")
full_model = LinearRegression().fit(X, y)
print("Full model coefficients: ", full_model.coef_)
# print("Full model intercept   : ", full_model.intercept_)

# ------------------------------------------------
# IBOSS Solver
# ------------------------------------------------
iboss = IBOSS_Solver(X=X, y=y, r=50)
iboss.solve()
print("IBOSS-estimated coefficients: ", iboss.coef_)
# print("IBOSS-estimated intercept   : ", iboss.intercept_)
# Indices of points chosen by IBOSS
chosen_indices = iboss.indices_
X_chosen = X[~chosen_indices]
y_chosen = y[~chosen_indices]

# ------------------------------------------------
# Blendenpik
blendenpik = BlendenpikSolver(X=X, y=y, r=50)
blendenpik.solve(method="unweighted", method_probs="shrinked")
print("Blendenpik-estimated coefficients: ", blendenpik.coef_)
X_chosen_blendenpik = X[blendenpik.indices_]
y_chosen_blendenpik = y[blendenpik.indices_]
print('########################################################')
# ------------------------------------------------

# MSE_train
print("Train error: ")
print("full model train error", np.average((full_model.predict(X) - y)**2))
print("iboss train error", np.average((iboss.predict(X) - y)**2))
print("blendenpik train error", np.average((blendenpik.predict(X) - y)**2))
print('########################################################')
#----------------------------------------------------------
# X_test
# X_test = np.random.normal(0, 1, X.shape)
x_1_test = np.random.uniform(0, 1, 500)
x_2_test = np.where((x_1 < 0.3) | (x_1 > 0.7), 
               x_1 + np.random.normal(0, 0.02, len(x_1)), 
               x_1)

max_x_2_first_segment_test = np.max(x_2_test[x_1_test < 0.3])
min_x_2_last_segment_test  = np.min(x_2_test[x_1_test > 0.7])
in_middle_test = (x_1 >= 0.3) & (x_1 <= 0.7)
x_2_test[in_middle_test] = np.random.uniform(max_x_2_first_segment_test, 
                                   min_x_2_last_segment_test, 
                                   size=np.sum(in_middle_test))
X_test = np.column_stack((x_1, x_2))

#----------------------------------------------------------
# y_test
y_test = X_test[:, 0] * beta[0] + X_test[:, 1] * beta[1] + np.random.normal(0, 0.5, len(X[:, 0]))
#----------------------------------------------------------

# MSE_test
print("Test error: ")
print("full model test error", np.average((full_model.predict(X_test) - y_test)**2))
print("iboss test error", np.average((iboss.predict(X_test) - y_test)**2))
print("blendenpik test error", np.average((blendenpik.predict(X_test) - y_test)**2))
print('########################################################')
# ------------------------------------------------
# Plotting
# ------------------------------------------------

# IBOSS comparison
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 1) Scatter entire dataset
ax.scatter(x_1, x_2, y, label='Všetky body', alpha=0.5)

# 2) Highlight chosen points in a different color/marker
ax.scatter(X_chosen[:, 0], 
           X_chosen[:, 1], 
           y_chosen, 
           label='IBOSS vybrané body',  
           s=50,
           alpha=0.5)

# 3) Plot regression surface
xx, yy = np.meshgrid(
    np.linspace(x_1.min(), x_1.max(), 20),
    np.linspace(x_2.min(), x_2.max(), 20)
)
zz_full = (full_model.intercept_
           + full_model.coef_[0] * xx
           + full_model.coef_[1] * yy)

zz_iboss = (iboss.intercept_ 
            + iboss.coef_[0] * xx 
            + iboss.coef_[1] * yy)

ax.plot_surface(xx, yy, zz_full, alpha=0.5, label='Rovina úplnej regresie')
ax.plot_surface(xx, yy, zz_iboss, alpha = 0.5, label="Rovina IBOSS regresie")

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('y')

plt.legend()
plt.show()

# ------------------------------------------------
# Blendenpik comparison

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 1) Scatter entire dataset
ax.scatter(x_1, x_2, y, label='Všetky body', alpha=0.5)

# 2) Highlight chosen points in a different color/marker
ax.scatter(X_chosen_blendenpik[:, 0], 
           X_chosen_blendenpik[:, 1], 
           y_chosen_blendenpik, 
           label='Vybrané body pomocou nezávislého výpočtu',  
           s=50,
           alpha=0.5)

# 3) Plot regression surface
xx, yy = np.meshgrid(
    np.linspace(x_1.min(), x_1.max(), 20),
    np.linspace(x_2.min(), x_2.max(), 20)
)

zz_blendenpik = (blendenpik.intercept_ 
            + blendenpik.coef_[0] * xx 
            + blendenpik.coef_[1] * yy)

ax.plot_surface(xx, yy, zz_full, alpha=0.5, label='Rovina úplnej regresie')
ax.plot_surface(xx, yy, zz_blendenpik, alpha = 0.5, label="Rovina regresie nezávislého výpočtu")

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('y')

plt.legend()
plt.show()
# ------------------------------------------------
# chosen points
ax = sns.scatterplot(x=x_1, y=x_2, color="skyblue", label="Všetky body")
sns.scatterplot(x=x_1[~iboss.indices_], y=x_2[~iboss.indices_], color="darkgreen", label="IBOSS body")
sns.scatterplot(x=x_1[blendenpik.indices_], y=x_2[blendenpik.indices_], color="orangered", label="Body zrazenej metódy")

common_indices = np.intersect1d(np.where(~iboss.indices_)[0], blendenpik.indices_)
sns.scatterplot(x=x_1[common_indices], y=x_2[common_indices], 
                color="black", label="Spoločné body IBOSS a zrazenej metódy")

# ax.set(xlabel="$x_1$", ylabel="$x_2$")
plt.show()
