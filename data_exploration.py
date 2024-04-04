import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("Stammdaten.csv")

# Select the two features
df_subset = df[['Werk OB', 'Kalender','Planlieferzeit Vertrag']]

df_subset['Kalender'].fillna('NaN', inplace=True)
# Display the first few rows of the dataframe
print(df_subset.head())

# Check for missing values
print(df_subset.isnull().sum())

# Plot distributions of each feature
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Werk OB', data=df_subset)
plt.title('Distribution of Werk OB')

plt.subplot(1, 2, 2)
sns.countplot(x='Kalender', data=df_subset, order=sorted(df_subset['Kalender'].dropna().unique()))
plt.title('Distribution of Kalender')

# sns.countplot(x='Planlieferzeit Vertrag', data=df_subset, order=df_subset['Planlieferzeit Vertrag'].value_counts().index).set_xlim(0,10)
# plt.title('Distribution of Planlieferzeit Vertrag')

plt.tight_layout()
plt.show()
kalender_categories = sorted(df_subset['Kalender'].unique())
# Plot relationship between features
# Plot relationship between features with sorted categories
plt.figure(figsize=(10, 6))
sns.countplot(x='Werk OB', hue='Kalender', data=df_subset, order=df_subset['Werk OB'].value_counts().index, hue_order=kalender_categories)
plt.title('Distribution of Kalender by Werk OB')
plt.legend(title='Kalender', loc='upper right')
plt.show()