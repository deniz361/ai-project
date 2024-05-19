import pandas as pd

# Basically what I am doing here is comparing Stammdaten and rule based anomalies to match the anomalies in Stammdaten.
# This way, we will be able to ceate a supervised dataset for anomalies.
# However, so far, there are no matching colums, whcih might be caused by either lack of matching data or wrong code logic 

def compare_csv_columns(file1, file2):
    # Read the CSV files into DataFrames with low_memory=False to handle mixed types
    df1 = pd.read_csv(file1, low_memory=False)
    df2 = pd.read_csv(file2, low_memory=False)

    # Get the columns from each DataFrame
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)

    # Find common columns
    common_columns = list(columns_df1.intersection(columns_df2))

    # Find columns unique to each DataFrame
    unique_to_df1 = list(columns_df1 - columns_df2)
    unique_to_df2 = list(columns_df2 - columns_df1)

    # Ensure both DataFrames contain only the common columns
    df1_common = df1[common_columns]
    df2_common = df2[common_columns]

    # Convert all columns to strings to handle any data type inconsistencies
    df1_common = df1_common.astype(str)
    df2_common = df2_common.astype(str)

    # Align DataFrames to have the same number of rows by intersecting indexes
    common_indexes = df1_common.index.intersection(df2_common.index)
    df1_common = df1_common.loc[common_indexes]
    df2_common = df2_common.loc[common_indexes]

    # Ensure both DataFrames have identical column orders by sorting columns
    df1_common = df1_common[sorted(df1_common.columns)]
    df2_common = df2_common[sorted(df2_common.columns)]

    # Ensure both DataFrames have identical index labels
    df1_common = df1_common.reset_index(drop=True)
    df2_common = df2_common.reset_index(drop=True)

    # Count rows with exactly matching values in the common columns
    exact_match_count = (df1_common.values == df2_common.values).all(axis=1).sum()

    # Return the results
    return {
        "common_columns": common_columns,
        "unique_to_file1": unique_to_df1,
        "unique_to_file2": unique_to_df2,
        "exact_match_count": exact_match_count
    }

# Path to CSV files
file1 = '/Users/awthura/THD/ai-project/datasets/rule_based_anomalies.csv'
file2 = '/Users/awthura/THD/ai-project/datasets/Stammdaten.csv'

result = compare_csv_columns(file1, file2)

print("Common columns:")
for col in result["common_columns"]:
    print(col)

print("\nColumns unique to file1")
for col in result["unique_to_file1"]:
    print(col)

print("\nColumns unique to file2")
for col in result["unique_to_file2"]:
    print(col)

print(f"\nNumber of rows with exact matches in common columns: {result['exact_match_count']}")
