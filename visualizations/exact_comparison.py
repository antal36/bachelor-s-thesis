import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid')

# Load your DataFrames
df1 = pd.read_csv(r"C:\Users\antal\Desktop\matfyz\bakalárka\scripts\solvers\all_results_0102_3.0.csv")
df2 = pd.read_csv(r"C:\Users\antal\Desktop\matfyz\bakalárka\scripts\solvers\all_results_0702.csv")

# Filter for the 'naive' solver
df1 = df1[df1["solver"] == "blendenpik"] 
df2 = df2[df2["solver"] == "blendenpik"]

# Add source labels
df1['source'] = '40_000 x 500'
df2['source'] = '20_000 x 500'

# Concatenate into one DataFrame
df_combined = pd.concat([df1, df2])

# Define metrics to compare
metrics = ['priemer_cas', 'odchylka_cas', 'MSE_train', 'MSE_test']

# Create subplots: 2 rows for 'weighted' and 'unweighted', and 3 columns for metrics
fig, axes = plt.subplots(nrows=2, ncols=len(metrics), figsize=(12, 10), sharex=False)

# Filter data for each row
methods = ['vážená', 'nevážená']

for row, method in enumerate(methods):
    filtered_df = df_combined[df_combined['method'] == method]
    
    for col, metric in enumerate(metrics):
        ax = axes[row, col]
        barplot = sns.barplot(
            data=filtered_df,
            x='prob method',
            y=metric,
            hue='source',
            ax=ax,
            errorbar=None
        )

        # Set x-axis tick labels as 'vplyvová' and 'zrazená'
        ax.set_xticklabels(['vplyvová', 'zrazená'])
        ax.set_xlabel("")

        # Display the y-axis label
        ax.set_ylabel(metric)

        # Hide y-ticks and y-axis line
        ax.set_yticks([])
        ax.tick_params(left=False)  # Disable ticks
        # ax.spines['left'].set_visible(False)  # Hide the y-axis line
        # ax.spines['top'].set_visible(False)  # Hide the y-axis line
        # ax.spines['right'].set_visible(False)  # Hide the y-axis line

        # Display values on bars
        for container in barplot.containers:
            barplot.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8)

        # Add row titles
        if col == 0:
            ax.set_title(f"{method.capitalize()} regresia", loc='left', fontsize=10, fontweight='bold')
        
        # Only keep legend on the last subplot
        if row == 0 and col == len(metrics) - 1:
            ax.legend(title='Dataset')
        else:
            ax.legend().remove()

# Adjust layout
plt.tight_layout()
plt.show()

