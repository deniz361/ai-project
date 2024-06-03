
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


categorical_columns = ["Material number", "Supplier", "Contract", "Contract Position", "Procurement type", 
                                    "Special procurement type", "Dispatcher", "Buyer", "Purchasing group", 
                                    "Purchasing lot size", "Calendar", "Plant", "Plant information record", 
                                    "Information record number", "Information record type",  "Product group",
                                    "Base unit"]
numerical_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value", 
                                  "Price unit", "Material master time", "Plant processing time"]
        


file_path = 'datasets/sampled_data.csv'

Data=Data_Preprocessing(file_path=file_path)
data, not_processed_data, categorical_columns_encoded= Data.preprocess_data_kmean()
print(data)


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

import pandas as pd
import numpy as np
from scipy import stats

# Assuming 'data' is your DataFrame containing numerical features

# Calculate mean and standard deviation for each numerical column
mean_values = data[numerical_columns].mean()
std_dev_values = data[numerical_columns].std()

# Calculate Z-scores for each numerical column
z_scores = (data[numerical_columns] - mean_values) / std_dev_values

# Optionally, you can add the Z-scores as new columns to the existing DataFrame
for col in numerical_columns:
    data[f'{col}_z_score'] = z_scores[col]


# Create a DataFrame to store the outlier flags
outlier_flags = pd.DataFrame()

# Apply outlier detection threshold (e.g., Z-score > 3) to identify outliers
for col in numerical_columns:
    outlier_flags[col + '_outlier'] = (z_scores[col] > 3).astype(int)

# Combine all outlier flags to a single anomaly label
data['anomaly_label'] = outlier_flags.max(axis=1)

# No need to drop outlier flags columns since they haven't been added yet
# Concatenate the anomaly label column with the input data
data = pd.concat([data, outlier_flags], axis=1)


# Combine all outlier flags to a single anomaly label
data['anomaly_label'] = data[[col + '_outlier' for col in numerical_columns]].max(axis=1)


features = data.drop(columns=[col + '_z_score' for col in numerical_columns])
# Remove columns related to anomaly detection from the input data

import xgboost as xgb
from sklearn.model_selection import train_test_split

# Assuming 'data' is your DataFrame containing numerical features and 'anomaly_label' is the binary target variable

# Split data into features and target variable
X = data.drop(columns=['anomaly_label'])
y = data['anomaly_label']

# Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic',enable_categorical=True, random_state=42)
xgb_model.fit(X, y)

# Predict anomalies on the testing data
y_pred_proba = xgb_model.predict_proba(X)[:, 1]  # Probability of being an outlier
y_pred = xgb_model.predict(X)  # Binary prediction (0 or 1)
data["xgbPossibility"]=y_pred_proba
# You can now use y_pred_proba or y_pred for further analysis or evaluation

# You can now use y_pred_proba or y_pred for further analysis or evaluation

data_without_nan = data.dropna(subset=['Total quantity_z_score'])

print(data['xgbPossibility'])
# Filter data for anomalies
# Filter not_processed_data for anomalies predicted by XGBoost
anomalies_xgboost = not_processed_data[data['xgbPossibility'] > 0.995]

# Save detected anomalies to a CSV file
anomalies_xgboost.to_csv('detected_anomalies_xgboost.csv', index=True)
print("saved xgb anomalies")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y, y_pred_proba))
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
#plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision_score(y, y_pred_proba))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
#plt.show()

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Normal', 'Anomaly'])
plt.yticks([0, 1], ['Normal', 'Anomaly'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
#plt.show()


# Feature Importance Plot (for top 5 features)
top_n = 5  # Specify the number of top features to plot
top_features_indices = xgb_model.feature_importances_.argsort()[-top_n:][::-1]
top_feature_names = X.columns[top_features_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features_indices)), xgb_model.feature_importances_[top_features_indices], align='center')
plt.yticks(range(len(top_features_indices)), top_feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title(f'Top {top_n} Feature Importance Plot')
#plt.show()

# Anomaly Score Distribution
plt.figure(figsize=(8, 6))
plt.hist(y_pred_proba, bins=50, color='blue', alpha=0.7)
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Anomaly Score Distribution')
#plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(X.iloc[:, 3], X.iloc[:, 4], c=y_pred, cmap='coolwarm', s=50, alpha=0.7)
plt.xlabel(X.columns[4])  # Use the first feature name as x-axis label
plt.ylabel(X.columns[3])  # Use the second feature name as y-axis label
plt.title('Scatter Plot with Anomaly Labels')
plt.colorbar()
#plt.show()

# Scatter Plot with Anomaly Labels
plt.figure(figsize=(12, 8))
plt.scatter(X.iloc[:, 0], X.iloc[:, 3], c=y_pred, cmap='coolwarm', s=50, alpha=0.7)
plt.xlabel(X.columns[0])  # Use the first feature name as x-axis label
plt.ylabel(X.columns[3])  # Use the second feature name as y-axis label
plt.title('Scatter Plot with Anomaly Labels')
plt.colorbar()
#plt.show()



import matplotlib.pyplot as plt

# Define a function to create scatter plots for anomalies
def visualize_anomalies(data, anomaly_labels):
    # Scatter plot for "Total quantity" column
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Total quantity'], data['Total value'], c=anomaly_labels, cmap='coolwarm')
    plt.xlabel('Total quantity')
    plt.ylabel('Total quantity')
    plt.title('Anomalies in Total Quantity')
    plt.colorbar(label='Anomaly Label')
    #plt\.show\(\)

# Assuming you have already predicted anomaly labels for the dataset
# Replace 'anomaly_labels' with your actual anomaly labels
visualize_anomalies(data, y_pred)

# Visualize statistical outliers in one of the numeric columns
plt.figure(figsize=(10, 6))
plt.hist(data_without_nan['Total quantity_z_score'],bins=50,  label='Z-scores')
plt.axvline(3, color='red', linestyle='dashed', linewidth=2, label='Outlier Threshold')
plt.title('Histogram of Z-scores for Total Inventory')
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.legend()
#plt\.show\(\)

import seaborn as sns


# Define a function to create scatter plots for anomalies
# def visualize_anomalies2(data, anomaly_labels):
#     # Pairplot for all numerical features
#     sns.pairplot(data, hue=da'anomaly_labels', palette={0: 'blue', 1: 'red'})
#     plt.title('Scatter Plot of Numerical Features with Anomalies')
#     #plt\.show\(\)

# # Visualize anomalies
# visualize_anomalies2(data[numerical_columns], y_pred)


# Visualize statistical outliers in one of the numeric columns
plt.figure(figsize=(10, 6))
plt.hist(data_without_nan['Total quantity_z_score'],  label='Z-scores')
plt.axvline(3, color='red', linestyle='dashed', linewidth=2, label='Outlier Threshold')
plt.title('Histogram of Z-scores for Total Inventory')
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.legend()
#plt\.show\(\)


# Find the top 5 columns with the most anomalies predicted by XGBoost
top_5_columns = xgb_model.feature_importances_.argsort()[-3:][::-1]

# Plot the distributions of the top 5 most anomalous columns
# for col_index in top_5_columns:
#     column_name = numerical_columns[col_index]
#     plt.figure(figsize=(10, 6))
#     plt.hist(data[column_name+'_z_score'], bins=50, label=f'Distribution of Z Score in {column_name}')
#     plt.title(f'Distribution of {column_name}')
#     plt.xlabel(column_name)
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.savefig(f'{column_name}_distribution.png')
#     #plt\.show\(\)
# Save the updated dataset
#data.to_csv('updated_with_anomalies_xgboost_stammdaten.csv', index=False)



import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt

# Handling missing values by replacing them with the median of each column
# for col in numerical_columns:
#     if data[col].isna().any():
#         data[col].fillna(data[col].median(), inplace=True)

# # Applying Z-score for anomaly detection in numeric columns
# for col in numerical_columns:
#     data[col + '_z_score'] = np.abs(stats.zscore(data[col]))
#     data[col + '_outlier'] = 0
#     data.loc[data[col + '_z_score'] > 3, col + '_outlier'] = 1  # Any Z-score > 3 is considered an outlier

# # Combine all outlier flags to a single anomaly label
# data['anomaly_label'] = data[[col + '_outlier' for col in numerical_columns]].max(axis=1)
X = data.drop(columns=['anomaly_label'])


X = pd.DataFrame(np.hstack((X[numerical_columns].values, categorical_columns_encoded)))
# Train Isolation Forest model for anomaly detection
from sklearn.impute import KNNImputer

# Instantiate KNN imputer
imputer = KNNImputer()

# Impute missing values using KNN imputation
X_imputed = imputer.fit_transform(X)
y = data['anomaly_label']


# Convert X_imputed into a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)



# Train Isolation Forest model for anomaly detection
if_model = IsolationForest(random_state=42)
if_model.fit(X_imputed,y)
not_processed_data=pd.DataFrame(not_processed_data)
# Predict anomaly labels using Isolation Forest
data2 = data.copy()
not_processed_data['if_anomaly'] = if_model.predict(X_imputed)
not_processed_data['if_anomaly'] = np.where(not_processed_data['if_anomaly'] == -1, 1, 0)  # Convert -1 to 1 for anomaly, 1 to 0 for normal
# Convert -1 to 1 for anomaly, 1 to 0 for normal
# Get anomaly scores for each sample
anomaly_scores = if_model.decision_function(X_imputed)

# Find the indices of the top 5 anomalies (lowest anomaly scores)
top_5_anomalies_indices = np.argsort(anomaly_scores)[:200]

# Get the corresponding rows (samples) from the original data
top_5_anomalies_data = not_processed_data.iloc[top_5_anomalies_indices]
# Save the top anomalies to a CSV file
top_5_anomalies_data.to_csv('top_150_Iforest_anomalies.csv', index=True)

#not_processed_data['if_anomaly'].to_csv('allanomaliIforest.csv', index=False)
# Find the top 5 columns with the most anomalies predicted by Isolation Forest
top_5_columns = np.argsort(np.sum(np.abs(if_model.decision_function(X_imputed))))

not_processed_data['if_anomaly'].iloc[top_5_anomalies_indices] = 1  # Convert anomalies to 1
not_processed_data['if_anomaly'].fillna(0, inplace=True)  # Fill non-anomalies with 0

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you already have X_imputed and if_model from your previous code

# Get anomaly scores for each sample
anomaly_scores = if_model.decision_function(X_imputed)

# Plot Histogram of Anomaly Scores
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()

# Create a scatter plot of anomalies
plt.figure(figsize=(10, 6))
plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=not_processed_data['if_anomaly'], cmap='coolwarm')
plt.title('Scatter Plot of Anomalies')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score')
plt.colorbar(label='Anomaly')
plt.grid(True)
#plt.show()

# Plot Feature Importance
# plt.figure(figsize=(10, 6))
# plt.barh(range(len(if_model.fe)), if_model.feature_importances_, align='center')
# plt.yticks(range(len(X.columns)), X.columns)
# plt.title('Feature Importance Plot')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()

# Box Plot of Anomaly Scores by Class
plt.figure(figsize=(10, 6))
sns.boxplot(x=not_processed_data['if_anomaly'], y=anomaly_scores)
plt.title('Box Plot of Anomaly Scores by Class')
plt.xlabel('Anomaly')
plt.ylabel('Anomaly Score')
plt.grid(True)
#plt.show()

# Correlation Heatmap
correlation_matrix = X_imputed.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
#plt.show()

# Plot the distributions of the top 5 most anomalous columns
for col_index in top_5_columns[:5]:
    column_name = numerical_columns[col_index]
    plt.figure(figsize=(10, 6))
    plt.hist(not_processed_data[column_name], bins=50, label=f'Distribution of {column_name}')
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{column_name}_distribution.png')
    #plt.show()

# Save the updated dataset
#data.to_csv('updated_with_anomalies_iforest_stammdaten.csv', index=True)



# Scatter plots for detected anomalies
plt.figure(figsize=(15, 10))
for col_index in range(len(numerical_columns)):
    column_name = numerical_columns[col_index]
    # Plot anomalies detected by XGBoost
    plt.scatter(not_processed_data[column_name][not_processed_data['if_anomaly'] == 1], not_processed_data[column_name][not_processed_data['if_anomaly'] == 1], c='red', label='Anomaly')
    # Plot normal not_processed_data points
    plt.scatter(not_processed_data[column_name][not_processed_data['if_anomaly'] == 0], not_processed_data[column_name][not_processed_data['if_anomaly'] == 0], c='blue', label='Normal')
    plt.xlabel(column_name)
    plt.ylabel('Value')
    plt.title(f'Scatter Plot of {column_name}')
    plt.legend()
    plt.savefig(f'{column_name}_scatter.png')
    #plt.show()


# Calculate ratios of detected anomalies on each column
anomaly_ratios = (data.filter(regex='_outlier$').sum() / len(data)).sort_values(ascending=False)
print(anomaly_ratios)

# Identify the most bizarre values of each column
# most_bizarre_values = {}
# for col in numerical_columns:
#     # Calculate Z-score for each column
#     z_scores = np.abs(stats.zscore(data[col]))
#     # Identify the most bizarre value
#     most_bizarre_value = data.loc[np.argmax(z_scores), col]
#     most_bizarre_values[col] = most_bizarre_value

# # Write the report to a text file
# with open("anomaly_report_iforest.txt", "w") as f:
#     f.write("Anomaly Ratios:\n")
#     for col, ratio in anomaly_ratios.items():
#         f.write(f"{col}: {ratio:.2f}\n")
#     f.write("\nMost Bizarre Values:\n")
#     for col, value in most_bizarre_values.items():
#         f.write(f"{col}: {value}\n")
#         f.write(f"Description: This value might occur due to ...\n\n")











# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from scipy import stats
# import matplotlib.pyplot as plt

# # Handling missing values by replacing them with the median of each column
# for col in numerical_columns:
#     if data[col].isna().any():
#         data[col].fillna(data[col].median(), inplace=True)

# # Applying Z-score for anomaly detection in numeric columns
# for col in numerical_columns:
#     data[col + '_z_score'] = np.abs(stats.zscore(data[col]))
#     data[col + '_outlier'] = 0
#     data.loc[data[col + '_z_score'] > 3, col + '_outlier'] = 1  # Any Z-score > 3 is considered an outlier

# # Combine all outlier flags to a single anomaly label
# data['anomaly_label'] = data[[col + '_outlier' for col in numerical_columns]].max(axis=1)

# # Train XGBoost model for anomaly detection
# xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
# xgb_model.fit(data[numerical_columns], data['anomaly_label'])

# # Predict anomaly labels using XGBoost
# data['xgb_anomaly'] = xgb_model.predict(data[numerical_columns])

# # Find the top 5 columns with the most anomalies predicted by XGBoost
# top_5_columns = xgb_model.feature_importances_.argsort()[-5:][::-1]

# # Plot the distributions of the top 5 most anomalous columns
# for col_index in top_5_columns:
#     column_name = numerical_columns[col_index]
#     plt.figure(figsize=(10, 6))
#     plt.hist(data[column_name], bins=50, label=f'Distribution of {column_name}')
#     plt.title(f'Distribution of {column_name}')
#     plt.xlabel(column_name)
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.savefig(f'{column_name}_distribution.png')
#     plt.show()

# # Save the updated dataset
# data.to_csv('updated_with_anomalies_xgboost_stammdaten.csv', index=False)



# # Scatter plots for detected anomalies
# plt.figure(figsize=(15, 10))
# for col_index in range(len(numerical_columns)):
#     column_name = numerical_columns[col_index]
#     # Plot anomalies detected by XGBoost
#     plt.scatter(data[column_name][data['xgb_anomaly'] == 1], data[column_name][data['xgb_anomaly'] == 1], c='red', label='Anomaly')
#     # Plot normal data points
#     plt.scatter(data[column_name][data['xgb_anomaly'] == 0], data[column_name][data['xgb_anomaly'] == 0], c='blue', label='Normal')
#     plt.xlabel(column_name)
#     plt.ylabel('Value')
#     plt.title(f'Scatter Plot of {column_name}')
#     plt.legend()
#     plt.savefig(f'{column_name}_scatter.png')
#     plt.show()



# # Calculate ratios of detected anomalies on each column
# anomaly_ratios = (data.filter(regex='_outlier$').sum() / len(data)).sort_values(ascending=False)

# # Identify the most bizarre values of each column
# most_bizarre_values = {}
# for col in numerical_columns:
#     # Calculate Z-score for each column
#     z_scores = np.abs(stats.zscore(data[col]))
#     # Identify the most bizarre value
#     most_bizarre_value = data.loc[np.argmax(z_scores), col]
#     most_bizarre_values[col] = most_bizarre_value

# # Write the report to a text file
# with open("anomaly_report.txt", "w") as f:
#     f.write("Anomaly Ratios:\n")
#     for col, ratio in anomaly_ratios.items():
#         f.write(f"{col}: {ratio:.2f}\n")
#     f.write("\nMost Bizarre Values:\n")
#     for col, value in most_bizarre_values.items():
#         f.write(f"{col}: {value}\n")
#         f.write(f"Description: This value might occur due to ...\n\n")











