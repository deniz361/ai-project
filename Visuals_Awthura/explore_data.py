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

file_path = 'Stammdaten.csv'
df = pd.read_csv(file_path, low_memory=False)

print_parameters(df)
corr_matrix(df)
plot_histograms(df, 'Planlieferzeit Vertrag', 'Vertrag Fix1')