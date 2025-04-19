import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme(style='whitegrid')

lambd: float = 0.8
x: np.ndarray = np.random.standard_t(6, 200)
y: np.ndarray = np.array([i + np.random.normal(0,1,1)[0] for i in x])
x_y_pairs: dict[np.float64, np.float64] = dict(zip(x,y))

fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 6))

H: np.ndarray = np.outer(x, x.T) * (1 / np.dot(x.T, x)) #hat matrix

weights: np.ndarray = np.diag(H) / np.sum(np.diag(H)) #weights for basic leveraging

weights_shrinked: np.ndarray = lambd * weights + (1-lambd)*(1/len(x)) #weights for shrinked leveraging

x_weight_pairs: dict[np.float64, np.float64] = dict(zip(x, weights)) #dictionaries for extracting weight for sampled x
x_weight_pairs_shrinked: dict[np.float64, np.float64] = dict(zip(x, weights_shrinked))

chosen_x: np.ndarray = np.random.choice(x, 20, replace=True, p=weights)
chosen_y: np.ndarray = np.array([x_y_pairs[point] for point in chosen_x])
chosen_weights: np.ndarray = np.array([x_weight_pairs[i] for i in chosen_x])

chosen_x_shrinked: np.ndarray = np.random.choice(x, 20, replace=True, p=weights_shrinked)
chosen_y_shrinked: np.ndarray = np.array([x_y_pairs[point] for point in chosen_x_shrinked])
chosen_weights_shrinked: np.ndarray = np.array([x_weight_pairs_shrinked[i] for i in chosen_x_shrinked])

for index in range(len(axs)):
    sns.scatterplot(x=x,y=y, ax=axs[index], color="blue") #scatter plot of generated x on axs

sns.scatterplot(x=chosen_x, y=chosen_y, ax=axs[1], color="orange")
sns.scatterplot(x=chosen_x_shrinked, y=chosen_y_shrinked, ax=axs[2], color="orange")

slope_all, intercept_all = np.polyfit(x=x, y=y, deg=1) #regression for all data
slope_sampled, intercept_sampled = np.polyfit(x=chosen_x, y=chosen_y, deg=1, w=chosen_weights) #regresion for basic leveraging
slope_sampled_shrinked, intercept_sampled_shrinked = np.polyfit(x=chosen_x_shrinked, y=chosen_y_shrinked, deg=1, w=chosen_weights_shrinked) #regression for shrinked leveraging

range_x: np.ndarray = np.linspace(min(x), max(x), 100)
y_fit_all: np.ndarray = slope_all * range_x + intercept_all
y_fit_sampled: np.ndarray = slope_sampled * range_x + intercept_sampled
y_fit_sampled_shrinked: np.ndarray = slope_sampled_shrinked * range_x + intercept_sampled_shrinked

sns.lineplot(x=range_x, y=y_fit_all, ax=axs[0], color="red")
sns.lineplot(x=range_x, y=y_fit_sampled, ax=axs[1], color="green")
sns.lineplot(x=range_x, y=y_fit_sampled_shrinked, ax=axs[2], color="purple")

axs[0].text(0.05, 0.95, f"$Sklon$: {slope_all:.2f}\n$Posun$: {intercept_all:.2f}", 
            transform=axs[0].transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
axs[1].text(0.05, 0.95, f"$Sklon$: {slope_sampled:.2f}\n$Posun$: {intercept_sampled:.2f}", 
            transform=axs[1].transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
axs[2].text(0.05, 0.95, f"$Sklon$: {slope_sampled_shrinked:.2f}\n$Posun$: {intercept_sampled_shrinked:.2f}", 
            transform=axs[2].transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

axs[0].set_title('Regresia so všetkými dátami')
axs[1].set_title('Regresia pomocou vplyvovej metódy')
axs[2].set_title('Regresia pomocou zrazenej vplyvovej metódy')

axs[0].set_xticklabels([])
axs[0].set_yticklabels([])
axs[1].set_xticklabels([])
axs[1].set_yticklabels([])
axs[2].set_xticklabels([])
axs[2].set_yticklabels([])

plt.show()



