import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    counts1, bins1, _ = plt.hist(df2[column1], bins=20, color='blue', alpha=0.7)
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

print_parameters(df)
corr_matrix(df)
plot_histograms(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
draw_box_plots(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
draw_violin_plots(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
draw_density_plots(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')
draw_scatter_plot(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')