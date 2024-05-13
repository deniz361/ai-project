import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import random
from itertools import combinations

def visualize_correlated_scatterplots(df):
    """
    Visualize correlated scatterplots with anomaly detection.

    Parameters:
    - df: pandas DataFrame containing the dataset with an "anomaly" column

    Returns:
    - None
    """
    corr_matrix = df.corr()
    correlated_features = corr_matrix[((corr_matrix > 0.5) | (corr_matrix < -0.5)) & (corr_matrix != 1.0)]
    num_pairs = len(correlated_features.stack())
    print(f"Number of scatter plots to be created: {num_pairs}")

    num_rows = 2
    num_cols = 3
    num_plots_per_page = num_rows * num_cols

    for i, ((feature1, feature2), correlation) in enumerate(correlated_features.stack().items(), 1):
        if (i - 1) % num_plots_per_page == 0:
            plt.figure(figsize=(15, 10))
        plt.subplot(num_rows, num_cols, i % num_plots_per_page or num_plots_per_page)

        # Scatter plot with anomalies colored differently
        sns.scatterplot(data=df, x=feature1, y=feature2, hue='anomaly', palette={1: 'red', 0: 'blue'}, legend=False)
        plt.title(f'Scatter Plot of {feature1} vs {feature2}\nCorrelation: {correlation:.2f}', fontsize=12)
        plt.xlabel(feature1, fontsize=10)
        plt.ylabel(feature2, fontsize=10)

        if i % num_plots_per_page == 0 or i == num_pairs:
            plt.tight_layout()
            plt.show()

def visualize_anomaly_heatmap(df, top_n=2, grid_size=10, correlation_threshold=0.5, graphs_per_page=4):
    """
    Visualize a heatmap of anomaly density for the top correlated feature pairs.

    Parameters:
    - df: pandas DataFrame containing the dataset with an "anomaly" column
    - top_n: Number of top correlated feature pairs to visualize (default=3)
    - grid_size: Number of cells in each dimension of the grid (default=10)
    - correlation_threshold: Threshold for considering feature pairs as correlated (default=0.5)
    - graphs_per_page: Maximum number of graphs to display per page (default=4)

    Returns:
    - None
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create a dictionary to store unique correlated pairs and their correlation values
    correlated_pairs = {}

    # Iterate over the columns of the correlation matrix
    for i, column1 in enumerate(corr_matrix.columns):
        for j, column2 in enumerate(corr_matrix.columns):
            if i < j:  # Only consider pairs where the column index of the first column is less than the second column
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > correlation_threshold:
                    # Store the pair and its correlation value in the dictionary
                    correlated_pairs[(column1, column2)] = correlation

    # Sort the dictionary by correlation value (absolute) in descending order
    sorted_correlated_pairs = dict(sorted(correlated_pairs.items(), key=lambda item: abs(item[1]), reverse=True))

    # Extract top correlated feature pairs
    top_correlated_pairs = list(sorted_correlated_pairs.keys())[:top_n]

    # Determine number of rows and columns for subplots
    num_rows = int(np.ceil(top_n / graphs_per_page))
    num_cols = min(top_n, graphs_per_page)

    # Calculate number of pages
    num_pages = int(np.ceil(top_n / graphs_per_page))

    processed_pairs = set()  # Set to store processed pairs

    for page in range(num_pages):
        # Create the figure and axes
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
        fig.suptitle(f'Anomaly Density Heatmaps (Page {page + 1})', fontsize=16)

        for i, ax in enumerate(axes.flatten()):
            plot_index = page * graphs_per_page + i
            if plot_index >= top_n:
                break

            feature1, feature2 = top_correlated_pairs[plot_index]

            x = df[feature1]
            y = df[feature2]

            # Create a grid
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)

            # Count anomalies in each cell of the grid
            anomaly_counts = np.zeros((grid_size, grid_size))
            for row in range(grid_size):
                for col in range(grid_size):
                    x_start, x_end = x_grid[row], x_grid[row+1] if row < grid_size-1 else x_max
                    y_start, y_end = y_grid[col], y_grid[col+1] if col < grid_size-1 else y_max
                    cell_anomalies = df[(df[feature1] >= x_start) & (df[feature1] < x_end) & 
                                        (df[feature2] >= y_start) & (df[feature2] < y_end)]['anomaly'].sum()
                    anomaly_counts[row, col] = cell_anomalies

            # Plot the heatmap
            sns.heatmap(anomaly_counts.T, cmap='Reds', annot=True, fmt='g', xticklabels=x_grid.round(2), yticklabels=y_grid.round(2),
                        ax=ax)
            ax.set_xlabel(feature1, fontsize=10)
            ax.set_ylabel(feature2, fontsize=10)
            ax.set_title(f'Anomaly Density Heatmap for {feature1} vs {feature2}', fontsize=12)

            processed_pairs.add((feature1, feature2))  # Add the processed pair to the set

        # Hide empty subplots42	0	0	350	66346	1	1	42	1

        for j in range(i + 1, num_rows * num_cols):
            axes.flatten()[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def visualize_anomaly_heatmap_side_by_side(df, n_pairs=1, grid_size=10):
    """
    Visualize anomaly density heatmaps side by side for random feature pairs.

    Parameters:
    - df: pandas DataFrame containing the dataset with an "anomaly" column
    - n_pairs: Number of random feature pairs to visualize (default=1)
    - grid_size: Number of cells in each dimension of the grid (default=10)

    Returns:
    - None
    """
    # Get feature columns
    features = df.columns[1:-1]  # Exclude the first and last columns

    # Get random pairs of features
    random_pairs = random.sample(list(combinations(features, 2)), n_pairs)

    # Calculate number of rows and columns for subplots
    num_rows = 1
    num_cols = 2

    # Calculate number of pages
    num_pages = int(np.ceil(n_pairs / num_cols))

    for page in range(num_pages):
        # Create the figure and axes
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
        fig.suptitle(f'Anomaly Density Heatmaps (Page {page + 1})', fontsize=16)

        for i, pair in enumerate(random_pairs[page*num_cols:page*num_cols+num_cols]):
            # Get the pair of features
            feature1, feature2 = pair

            # Extract data for the pair of features
            x = df[feature1]
            y = df[feature2]

            # Create a grid
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)

            # Count anomalies and data points in each cell of the grid
            anomaly_counts = np.zeros((grid_size, grid_size))
            data_counts = np.zeros((grid_size, grid_size))
            for row in range(grid_size):
                for col in range(grid_size):
                    x_start, x_end = x_grid[row], x_grid[row+1] if row < grid_size-1 else x_max
                    y_start, y_end = y_grid[col], y_grid[col+1] if col < grid_size-1 else y_max
                    cell_anomalies = ((df[feature1] >= x_start) & (df[feature1] < x_end) & 
                                       (df[feature2] >= y_start) & (df[feature2] < y_end) & 
                                       (df['anomaly'] == 1)).sum()
                    anomaly_counts[row, col] = cell_anomalies
                    cell_data = ((df[feature1] >= x_start) & (df[feature1] < x_end) & 
                                 (df[feature2] >= y_start) & (df[feature2] < y_end)).sum()
                    data_counts[row, col] = cell_data

            # Plot the heatmaps side by side
            sns.heatmap(anomaly_counts.T, cmap='Reds', annot=True, fmt='g', cbar=False, ax=axes[0], annot_kws={"size": 8})
            sns.heatmap(data_counts.T, cmap='Blues', annot=True, fmt='g', cbar=False, ax=axes[1], annot_kws={"size": 8})

            # Set titles and labels
            axes[0].set_title('Anomaly Density', fontsize=12)
            axes[1].set_title('All Data Points', fontsize=12)
            axes[0].set_xlabel(feature1, fontsize=10)
            axes[0].set_ylabel(feature2, fontsize=10)
            axes[1].set_xlabel(feature1, fontsize=10)
            axes[1].set_ylabel(feature2, fontsize=10)

        # Hide empty subplots
        for j in range(i + 1, num_rows * num_cols):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def print_anomaly_summary(df):
    """
    Print the total number of columns, anomalies, and non-anomalies in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing the dataset with an "anomaly" column

    Returns:
    - None
    """
    total_columns = len(df.columns)
    total_anomalies = df['anomaly'].sum()
    total_non_anomalies = len(df) - total_anomalies

    print(f"Total number of columns: {total_columns}")
    print(f"Total number of anomalies: {total_anomalies}")
    print(f"Total number of non-anomalies: {total_non_anomalies}")




file_path = 'sampled_data_with_anomalies.csv'
df = pd.read_csv(file_path, low_memory=False)
print_anomaly_summary(df)
visualize_correlated_scatterplots(df)
#visualize_anomaly_density(df)
#visualize_anomaly_heatmap(df)
#visualize_anomaly_heatmap_side_by_side(df)