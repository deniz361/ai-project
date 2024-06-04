import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df1 = pd.read_csv('anomalies/anomalies_kmean.csv')
df2 = pd.read_csv('anomalies/dbscan_labeled_data.csv')
df3 = pd.read_csv('anomalies/detected_anomalies_xgboost.csv')
df4 = pd.read_csv('anomalies/lof_with_anomalies.csv')
df5 = pd.read_csv('anomalies/rfc_anomaly_labels.csv')
df6 = pd.read_csv('anomalies/top_150_Iforest_anomalies.csv')

df1.columns = ['kmean_' + col for col in df1.columns]
df2.columns = ['dbscan_' + col for col in df2.columns]
df3.columns = ['xgboost_' + col for col in df3.columns]
df4.columns = ['lof_' + col for col in df4.columns]
df5.columns = ['rfc_' + col for col in df5.columns]
df6.columns = ['iforest_' + col for col in df6.columns]

df = pd.concat([df1, df2, df3, df4, df5, df6], axis=1)

df.to_csv("anomalies/all_anomalies.csv")

# Fill NaN values in 'outlier_features' column with an empty string
df['outlier_features'] = df['outlier_features'].fillna('').apply(lambda x: x.split(' '))

# Group by algorithm and count the occurrence of each outlier feature
algorithm_outlier_counts = df.explode('outlier_features').groupby(['algorithm', 'outlier_features']).size().reset_index(name='count')

# Pivot the data to visualize algorithms vs outlier features
pivot_data = algorithm_outlier_counts.pivot(index='algorithm', columns='outlier_features', values='count').fillna(0)

# Plot heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(pivot_data, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title('Algorithms vs Outlier Features')
plt.xlabel('Outlier Features')
plt.ylabel('Algorithm')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Save the plot
plt.savefig('algorithm_outlier_features_heatmap.png')

# Show the plot
plt.show()

# Find common rows of data based on the UUID
common_rows = df[df.duplicated(subset='uuid', keep=False)]
print('Common Rows of Data based on UUID:')
print(common_rows[['uuid', 'algorithm']])
