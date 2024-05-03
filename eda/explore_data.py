import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def print_parameters(df):

    print("Top 5 rows:")
    print(df.head())

    print("Number of rows in the dataset:", len(df))
    print("Number of columns in the dataset:", df.shape[1])

def corr_matrix(df):

    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.3)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


def plot_histograms(df, column1, column2):
    df2 = df[[column1, column2]].copy()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    counts1, bins1, _ = plt.hist(df2[column1], bins=3, color='blue', alpha=0.7)
    plt.title(f'Histogram of {column1}')
    plt.xlabel(column1)
    plt.ylabel('Frequency')
 


    plt.subplot(1, 2, 2)
    counts2, bins2, _ = plt.hist(df2[column2], bins=20, color='green', alpha=0.7)
    plt.title(f'Histogram of {column2}')
    plt.xlabel(column2)
    plt.ylabel('Frequency')


    plt.tight_layout()
    plt.show()

def draw_box_plots(df, column1, column2):

    df2 = df[[column1, column2]].copy()

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df2[[column1]], orient='v', color='skyblue', showfliers=True, whis=1.5)
    plt.title(f'Box Plot of {column1}')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df2[[column2]], orient='v', color='lightgreen', showfliers=True, whis=1.5)
    plt.title(f'Box Plot of {column2}')

    plt.tight_layout()
    plt.show()

def draw_violin_plots(df, column1, column2):

    df2 = df[[column1, column2]].copy()

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    sns.violinplot(data=df2[[column1]], orient='v', color='skyblue')
    plt.title(f'Violin Plot of {column1}')

    plt.subplot(1, 2, 2)
    sns.violinplot(data=df2[[column2]], orient='v', color='lightgreen')
    plt.title(f'Violin Plot of {column2}')

    plt.tight_layout()

    plt.show()
 
def draw_Mult_violin_plots(df):
    # Get the list of column names
    columns = df.columns
    
    # Determine the number of rows and columns for the subplot grid
    num_plots_per_page = 4
    num_rows = math.ceil(num_plots_per_page / 2)
    num_cols = 2

    # Iterate over features and create violin plots
    for i in range(0, len(columns), num_plots_per_page):
        # Create a new figure for every 6 plots
        plt.figure(figsize=(15, 10))
        
        # Iterate over the next 6 features
        for j, column in enumerate(columns[i:i+num_plots_per_page], 1):
            plt.subplot(num_rows, num_cols, j)
            sns.violinplot(data=df[[column]], orient='v', color='lime')
            plt.title(f'Violin Plot of {column}',fontsize=8)

        plt.tight_layout()
        plt.show()   

def draw_density_plots(df, column1, column2):
    df2 = df[[column1, column2]].copy()

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df2[column1], color='skyblue', fill=True)
    plt.title(f'Density Plot of {column1}')

    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df2[column2], color='lightgreen', fill=True)
    plt.title(f'Density Plot of {column2}')

    plt.tight_layout()

    plt.show()

def draw_scatter_plot(df, column1, column2):

    df2 = df[[column1, column2]].copy()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df2, x=column1, y=column2, color='blue')
    plt.title(f'Scatter Plot of {column1} vs {column2}')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.tight_layout()
    plt.show()

file_path = 'Stammdaten.csv'
df = pd.read_csv(file_path, low_memory=False)

def plot_top10_common_values(df, column1, column2):

    plt.figure(figsize=(15, 10))
    
    df2 = df[[column1, column2]].copy()

    for i, column in enumerate(df2.columns, 1):

        top10_values = df2[column].value_counts().head(10).sort_index()

        plt.subplot(2, 3, i)

        sns.countplot(data=df, x=column, order=top10_values.index, palette='viridis')

        plt.title(f'Top 10 Unique Values for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        
        for j, value in enumerate(top10_values):
            plt.text(j, value + 10, value, ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()

    plt.show()

def visualize_correlated_scatterplots(df):
    
    corr_matrix = df.corr()
    correlated_features = corr_matrix[((corr_matrix > 0.3) | (corr_matrix < -0.3)) & (corr_matrix != 1.0)]
    num_pairs = len(correlated_features.stack())
    print(f"Number of scatter plots to be created: {num_pairs}")

    num_rows = 2
    num_cols = 3
    num_plots_per_page = num_rows * num_cols

    for i, ((feature1, feature2), correlation) in enumerate(correlated_features.stack().items(), 1):
        if (i - 1) % num_plots_per_page == 0:
            plt.figure(figsize=(15, 10))
        plt.subplot(num_rows, num_cols, i % num_plots_per_page or num_plots_per_page)

        sns.scatterplot(data=df, x=feature1, y=feature2)
        plt.title(f'Scatter Plot of {feature1} vs {feature2}\nCorrelation: {correlation:.2f}', fontsize=12)
        plt.xlabel(feature1, fontsize=10)
        plt.ylabel(feature2, fontsize=10)

        if i % num_plots_per_page == 0 or i == num_pairs:
            plt.tight_layout()
            plt.show()


#corr_matrix(df)
#visualize_correlated_scatterplots(df)
# plot_histograms(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
# draw_box_plots(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
# draw_violin_plots(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
draw_Mult_violin_plots(df)
# draw_density_plots(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
# draw_scatter_plot(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
# plot_top10_common_values(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')