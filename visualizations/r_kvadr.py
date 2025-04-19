import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns

# sns.set_theme(style="whitegrid")

fig, axs = plt.subplots(1,2, sharex=True, sharey=True)


x = np.random.normal(0, 1, 100)

y_1 = np.array([i + np.random.normal(0, 0.3, 1)[0] for i in x])
y_2 = np.array([i + np.random.normal(0, 1, 1)[0] for i in x])

slope1, intercept1 = np.polyfit(x=x, y=y_1, deg=1)
slope2, intercept2 = np.polyfit(x=x, y=y_2, deg=1)

x_range = np.linspace(min(x), max(x), 100)
y_1_predicted = np.array([slope1 * i + intercept1 for i in x_range])
y_2_predicted = np.array([slope2 * i + intercept2 for i in x_range])
sns.scatterplot(x=x, y=y_1, ax=axs[0])
sns.scatterplot(x=x, y=y_2, ax=axs[1])
sns.lineplot(x=x_range, y=y_1_predicted, color="red", ax=axs[0])
sns.lineplot(x=x_range, y=y_2_predicted, color="red", ax=axs[1])
r_2_1 = round(r2_score(y_1, [slope1 * i + intercept1 for i in x]), 2)
r_2_2 = round(r2_score(y_2, [slope2 * i + intercept2 for i in x]), 2)


axs[0].text(0.05, 0.95, f'$R^2 = {r_2_1}$', transform=axs[0].transAxes, 
            fontsize=12, verticalalignment='top')
axs[1].text(0.05, 0.95, f'$R^2 = {r_2_2}$', transform=axs[1].transAxes, 
            fontsize=12, verticalalignment='top')

axs[0].set_xticklabels([])
axs[0].set_yticklabels([])
axs[1].set_xticklabels([])
axs[1].set_yticklabels([])

fig.suptitle("Hodnota $R^2$ pre dáta s rôznym rozptylom")
plt.show()
