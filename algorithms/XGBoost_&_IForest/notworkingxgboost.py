
import numpy as np
import pandas as pd
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
parent_dir2 = os.path.dirname(parent_dir)

sys.path.append(parent_dir2)
from datasets.data_preprocessing import Data_Preprocessing

file_path="datasets/Stammdaten.csv"
Data=Data_Preprocessing(file_path=file_path)
data= Data.preprocess_data()
print(data)

categorical_columns = ["Material number", "Supplier", "Contract", "Contract Position", "Procurement type", 
                                    "Special procurement type", "Dispatcher", "Buyer", "Purchasing group", 
                                    "Purchasing lot size", "Calendar", "Plant", "Plant information record", 
                                    "Information record number", "Information record type",  "Product group",
                                    "Base unit"]
numerical_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value", 
                                  "Price unit", "Plant processing time", "Material master time"]
        


# data[categorical_columns] = data[categorical_columns].astype('category')
# data[numerical_columns] = data[numerical_columns].astype('int64')





import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt

# Encode categorical variables
#categorical_columns = data.select_dtypes(include=['object']).columns
#data[categorical_columns] = data[categorical_columns].apply(lambda col: pd.factorize(col)[0])


# Handling missing values by replacing them with the median of each column
# for col in numerical_columns:
#     if data[col].isna().any():
#         data[col].fillna(data[col].median(), inplace=True)

# Applying Z-score for anomaly detection in numeric columns
for col in numerical_columns:
    data[col + '_z_score'] = np.abs(stats.zscore(data[col]))
    data[col + '_outlier'] = 0
    data.loc[data[col + '_z_score'] > 3, col + '_outlier'] = 1  # Any Z-score > 3 is considered an outlier

# Combine all outlier flags to a single anomaly label
data['anomaly_label'] = data[[col + '_outlier' for col in numerical_columns]].max(axis=1)

# Train XGBoost model for anomaly detection
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
xgb_model.fit(data[numerical_columns], data['anomaly_label'])

# Predict anomaly labels using XGBoost
data['xgb_anomaly'] = xgb_model.predict(data[numerical_columns])

data_without_nan = data.dropna(subset=['Total quantity_z_score'])


# Visualize statistical outliers in one of the numeric columns
plt.figure(figsize=(10, 6))
plt.hist(data_without_nan['Total quantity_z_score'],  label='Z-scores')
plt.axvline(3, color='red', linestyle='dashed', linewidth=2, label='Outlier Threshold')
plt.title('Histogram of Z-scores for Total Inventory')
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Save the updated dataset
data.to_csv('updated_with_anomalies_xgboost_stammdaten.csv', index=False)



import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt

# Handling missing values by replacing them with the median of each column
for col in numerical_columns:
    if data[col].isna().any():
        data[col].fillna(data[col].median(), inplace=True)

# Applying Z-score for anomaly detection in numeric columns
for col in numerical_columns:
    data[col + '_z_score'] = np.abs(stats.zscore(data[col]))
    data[col + '_outlier'] = 0
    data.loc[data[col + '_z_score'] > 3, col + '_outlier'] = 1  # Any Z-score > 3 is considered an outlier

# Combine all outlier flags to a single anomaly label
data['anomaly_label'] = data[[col + '_outlier' for col in numerical_columns]].max(axis=1)

# Train Isolation Forest model for anomaly detection
if_model = IsolationForest(random_state=42)
if_model.fit(data[numerical_columns])

# Predict anomaly labels using Isolation Forest
data['if_anomaly'] = if_model.predict(data[numerical_columns])
data['if_anomaly'] = np.where(data['if_anomaly'] == -1, 1, 0)  # Convert -1 to 1 for anomaly, 1 to 0 for normal

# Find the top 5 columns with the most anomalies predicted by Isolation Forest
top_5_columns = np.argsort(np.sum(np.abs(if_model.decision_function(data[numerical_columns]))))

# Plot the distributions of the top 5 most anomalous columns
for col_index in top_5_columns[:5]:
    column_name = numerical_columns[col_index]
    plt.figure(figsize=(10, 6))
    plt.hist(data[column_name], bins=50, label=f'Distribution of {column_name}')
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{column_name}_distribution.png')
    plt.show()

# Save the updated dataset
data.to_csv('updated_with_anomalies_iforest_stammdaten.csv', index=False)



# Scatter plots for detected anomalies
plt.figure(figsize=(15, 10))
for col_index in range(len(numerical_columns)):
    column_name = numerical_columns[col_index]
    # Plot anomalies detected by XGBoost
    plt.scatter(data[column_name][data['if_anomaly'] == 1], data[column_name][data['if_anomaly'] == 1], c='red', label='Anomaly')
    # Plot normal data points
    plt.scatter(data[column_name][data['if_anomaly'] == 0], data[column_name][data['if_anomaly'] == 0], c='blue', label='Normal')
    plt.xlabel(column_name)
    plt.ylabel('Value')
    plt.title(f'Scatter Plot of {column_name}')
    plt.legend()
    plt.savefig(f'{column_name}_scatter.png')
    plt.show()


# Calculate ratios of detected anomalies on each column
anomaly_ratios = (data.filter(regex='_outlier$').sum() / len(data)).sort_values(ascending=False)

# Identify the most bizarre values of each column
most_bizarre_values = {}
for col in numerical_columns:
    # Calculate Z-score for each column
    z_scores = np.abs(stats.zscore(data[col]))
    # Identify the most bizarre value
    most_bizarre_value = data.loc[np.argmax(z_scores), col]
    most_bizarre_values[col] = most_bizarre_value

# Write the report to a text file
with open("anomaly_report_iforest.txt", "w") as f:
    f.write("Anomaly Ratios:\n")
    for col, ratio in anomaly_ratios.items():
        f.write(f"{col}: {ratio:.2f}\n")
    f.write("\nMost Bizarre Values:\n")
    for col, value in most_bizarre_values.items():
        f.write(f"{col}: {value}\n")
        f.write(f"Description: This value might occur due to ...\n\n")











import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt

# Handling missing values by replacing them with the median of each column
for col in numerical_columns:
    if data[col].isna().any():
        data[col].fillna(data[col].median(), inplace=True)

# Applying Z-score for anomaly detection in numeric columns
for col in numerical_columns:
    data[col + '_z_score'] = np.abs(stats.zscore(data[col]))
    data[col + '_outlier'] = 0
    data.loc[data[col + '_z_score'] > 3, col + '_outlier'] = 1  # Any Z-score > 3 is considered an outlier

# Combine all outlier flags to a single anomaly label
data['anomaly_label'] = data[[col + '_outlier' for col in numerical_columns]].max(axis=1)

# Train XGBoost model for anomaly detection
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
xgb_model.fit(data[numerical_columns], data['anomaly_label'])

# Predict anomaly labels using XGBoost
data['xgb_anomaly'] = xgb_model.predict(data[numerical_columns])

# Find the top 5 columns with the most anomalies predicted by XGBoost
top_5_columns = xgb_model.feature_importances_.argsort()[-5:][::-1]

# Plot the distributions of the top 5 most anomalous columns
for col_index in top_5_columns:
    column_name = numerical_columns[col_index]
    plt.figure(figsize=(10, 6))
    plt.hist(data[column_name], bins=50, label=f'Distribution of {column_name}')
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{column_name}_distribution.png')
    plt.show()

# Save the updated dataset
data.to_csv('updated_with_anomalies_xgboost_stammdaten.csv', index=False)









# Scatter plots for detected anomalies
plt.figure(figsize=(15, 10))
for col_index in range(len(numerical_columns)):
    column_name = numerical_columns[col_index]
    # Plot anomalies detected by XGBoost
    plt.scatter(data[column_name][data['xgb_anomaly'] == 1], data[column_name][data['xgb_anomaly'] == 1], c='red', label='Anomaly')
    # Plot normal data points
    plt.scatter(data[column_name][data['xgb_anomaly'] == 0], data[column_name][data['xgb_anomaly'] == 0], c='blue', label='Normal')
    plt.xlabel(column_name)
    plt.ylabel('Value')
    plt.title(f'Scatter Plot of {column_name}')
    plt.legend()
    plt.savefig(f'{column_name}_scatter.png')
    plt.show()



# Calculate ratios of detected anomalies on each column
anomaly_ratios = (data.filter(regex='_outlier$').sum() / len(data)).sort_values(ascending=False)

# Identify the most bizarre values of each column
most_bizarre_values = {}
for col in numerical_columns:
    # Calculate Z-score for each column
    z_scores = np.abs(stats.zscore(data[col]))
    # Identify the most bizarre value
    most_bizarre_value = data.loc[np.argmax(z_scores), col]
    most_bizarre_values[col] = most_bizarre_value

# Write the report to a text file
with open("anomaly_report.txt", "w") as f:
    f.write("Anomaly Ratios:\n")
    for col, ratio in anomaly_ratios.items():
        f.write(f"{col}: {ratio:.2f}\n")
    f.write("\nMost Bizarre Values:\n")
    for col, value in most_bizarre_values.items():
        f.write(f"{col}: {value}\n")
        f.write(f"Description: This value might occur due to ...\n\n")











