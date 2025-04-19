import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random

x = np.random.standard_t(7, 500)
y = np.array([element + np.random.normal(0, 2, 1)[0] for element in x])
pairs = dict(zip(x, y))

x_sampled = np.array(random.choices(x, weights=[1/len(x) for _ in range(len(x))], k=15))
y_sampled = np.array([pairs[i] for i in x_sampled])

sns.scatterplot(x=x, y=y)
sns.scatterplot(x=x_sampled, y=y_sampled)

slope, intercept = np.polyfit(x, y, 1)

x_range = np.linspace(min(x), max(x), 100)
y_fit_full = slope * x_range + intercept  

sns.lineplot(x=x_range, y=y_fit_full, color='red', label='Úplná lineárna regresia', errorbar=None)

slope_sampled, intercept_sampled = np.polyfit(x_sampled, y_sampled, 1)
y_fit_sampled = slope_sampled * x_range + intercept_sampled  # Calculate the y-values for sampled data

sns.lineplot(x=x_range, y=y_fit_sampled, color='green', label='Vzorkovaná lineárna regresia', errorbar=None)

# plt.grid(visible=True)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.legend()
plt.show()