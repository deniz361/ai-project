import pandas as pd

# Load the CSV files
file1 = 'detected_anomalies_xgboost_with_Index.csv'
file2 = 'top_150_Iforest_anomalies_with_Index.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge the dataframes on the index column
merged = pd.merge(df1, df2, on='Index')

# Save rows with the same index to a new file
merged.to_csv('rows_with_same_index.csv', index=False)

# Concatenate the dataframes
combined = pd.concat([df1, df2])

# Drop duplicate rows
combined_no_duplicates = combined.drop_duplicates()

# Save the combined file without duplicates
combined_no_duplicates.to_csv('combined_no_duplicates.csv', index=False)
