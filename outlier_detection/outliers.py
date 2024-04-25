import matplotlib.pyplot as plt
import pandas as pd


def column_description(df):
    """
    Provides a short description of every column in the dataset.

    Args:
    - df: Pandas DataFrame

    Returns:
    - None (prints column descriptions)
    """
    for col in df.columns:
        print(f"Column: {col}")
        print("Data type:", df[col].dtype)
        print("Number of unique values:", df[col].nunique())
        if df[col].dtype == 'object':
            print("Sample values:", df[col].unique()[:5])
        else:
            print("Summary statistics:")
            print(df[col].describe())
        print("-------------------------------------")


def plot_all_columns(df):
    """
    Plots all columns of the DataFrame on a 5 by 5 grid, using barplots
    for categorical columns and boxplots for numeric columns.

    Args:
    - df: Pandas DataFrame

    Returns:
    - None (displays the plots)
    """
    num_columns = len(df.columns)
    num_rows = (num_columns + 4) // 5  # Ceiling division to determine the number of rows

    fig, axes = plt.subplots(num_rows, 5, figsize=(20, 4 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]
        if df[col].dtype == 'object':
            value_counts = df[col].value_counts()
            ax.bar(value_counts.index, value_counts.values, width=0.8)
            ax.set_title(f"Barplot of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.boxplot(df[col].dropna(), vert=False)
            ax.set_title(f"Boxplot of {col}")
            ax.set_xlabel(col)

    # Hide any remaining empty subplots
    for i in range(num_columns, num_rows * 5):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# Read dataset
df = pd.read_csv('../Stammdaten.csv', low_memory=False)
df = df.sample(frac=0.25, random_state=42)

# Change falsely recognized datatypes
df['Lieferant OB'] = df['Lieferant OB'].astype('object')
df['Vertrag OB'] = df['Vertrag OB'].astype('object')
df['Vertragsposition OB'] = df['Vertragsposition OB'].astype('object')
df['Planlieferzeit Vertrag'] = df['Planlieferzeit Vertrag'].astype('object')
df['Vertrag Fix1'] = df['Vertrag Fix1'].astype('object')
df['Vertrag_Fix2'] = df['Vertrag_Fix2'].astype('object')
df['Sonderbeschaffungsart'] = df['Sonderbeschaffungsart'].astype('object')
df['Disponent'] = df['Disponent'].astype('object')
df['Einkäufer'] = df['Einkäufer'].astype('object')
df['Preiseinheit'] = df['Preiseinheit'].astype('object')
df['Werk OB'] = df['Werk OB'].astype('object')
df['Werk Infosatz'] = df['Werk Infosatz'].astype('object')
df['Infosatznummer'] = df['Infosatznummer'].astype('object')
df['Infosatztyp'] = df['Infosatztyp'].astype('object')
df['WE-Bearbeitungszeit'] = df['WE-Bearbeitungszeit'].astype('object')


print(df.shape)
df = df[df['Gesamtbestand'] != 0]
print(df.shape)
df = df[df['Gesamtbestand'] < 5_000_000]
print(df.shape)
df = df[df['Gesamtwert'] < 6_000_000]
print(df.shape)


# Get column descriptions
column_description(df)

# Plot all columns on a 5 by 5 grid
plot_all_columns(df)
