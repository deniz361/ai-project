
import numpy as np
import pandas as pd
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
parent_dir2 = os.path.dirname(parent_dir)

sys.path.append(parent_dir2)
from datasets.data_preprocessing import Data_Preprocessing
from algorithms.RFC import rfc_new

categorical_columns = ["Material number", "Supplier", "Contract", "Contract Position", "Procurement type", 
                                    "Special procurement type", "Dispatcher", "Buyer", "Purchasing group", 
                                    "Purchasing lot size", "Calendar", "Plant", "Plant information record", 
                                    "Information record number", "Information record type",  "Product group",
                                    "Base unit"]
numerical_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value", 
                                  "Price unit", "Material master time", "Plant processing time"]
numeric_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value",
                       "Price unit", "Plant processing time", "Material master time"]
train_filepath = 'datasets/split/supervised_learning_dataset/train.csv'
val_filepath = 'datasets/split/supervised_learning_dataset/val.csv'

X_train, Y_train, x_val, Y_val = rfc_new.load_data(train_filepath, val_filepath)
x_val2=x_val
file_path = 'datasets/split/supervised_learning_dataset/train.csv'

Data=Data_Preprocessing(file_path=file_path)
data, not_processed_data, categorical_columns_encoded= Data.preprocess_data_kmean()

#data=pd.DataFrame(X_train,columns=numeric_columns)

Data=Data_Preprocessing(file_path=val_filepath)
val_data, not_processed_data, categorical_columns_encoded= Data.preprocess_data_kmean()

data=X_train

# Encode categorical variables
#categorical_columns = data.select_dtypes(include=['object']).columns
#data[categorical_columns] = data[categorical_columns].apply(lambda col: pd.factorize(col)[0])
# print(data.colums)


# Handling missing values by replacing them with the median of each column


# Applying Z-score for anomaly detection in numeric columns

import pandas as pd
import numpy as np
from scipy import stats

# Assuming 'data' is your DataFrame containing numerical features


#x_val= x_val.drop(columns=['anomaly'])
# Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic',enable_categorical=True, random_state=42)
xgb_model.fit(X_train, Y_train)


Y_pred_train = xgb_model.predict(X_train)
Y_pred_val = xgb_model.predict(x_val2)

print("Training Accuracy:", accuracy_score(Y_train, Y_pred_train))
print("Validation Accuracy:", accuracy_score(Y_val, Y_pred_val))
print("\nValidation Classification Report:\n", classification_report(Y_val, Y_pred_val))
print("\nValidation Confusion Matrix:\n", confusion_matrix(Y_val, Y_pred_val))

# Predict anomalies on the testing data
y_pred_proba = xgb_model.predict_proba(X_train)[:, 1]  # Probability of being an outlier
y_pred = xgb_model.predict(X_train)  # Binary prediction (0 or 1)
data["xgbPossibility"]=y_pred_proba
# You can now use y_pred_proba or y_pred for further analysis or evaluation

# You can now use y_pred_proba or y_pred for further analysis or evaluation

#data_without_nan = data.dropna(subset=['Total quantity_z_score'])

print(data['xgbPossibility'])
# Filter data for anomalies
# Filter not_processed_data for anomalies predicted by XGBoost
anomalies_xgboost = data[data['xgbPossibility'] > 0.995]


# feature_importances = xgb_model.feature_importances_

# # Create a DataFrame to display feature importances
# feature_importances_df = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': feature_importances
# })
# feature_importances_df.to_csv('feature_importances_xgb.csv')

# Save detected anomalies to a CSV file
anomalies_xgboost.to_csv('detected_anomalies_xgboost.csv', index=True)
print("saved xgb anomalies")

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score

# ROC Curve
fpr, tpr, _ = roc_curve(Y_train, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(Y_train, y_pred_proba))
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
#plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(Y_train, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision_score(Y_train, y_pred_proba))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
#plt.show()

# Confusion Matrix
cm = confusion_matrix(Y_train, y_pred)
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
top_n = 8  # Specify the number of top features to plot
top_features_indices = xgb_model.feature_importances_.argsort()[-top_n:][::-1]
top_feature_names = X_train.columns[top_features_indices]

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
plt.scatter(X_train.iloc[:, 3], X_train.iloc[:, 4], c=y_pred, cmap='coolwarm', s=50, alpha=0.7)
plt.xlabel(X_train.columns[4])  # Use the first feature name as x-axis label
plt.ylabel(X_train.columns[3])  # Use the second feature name as y-axis label
plt.title('Scatter Plot with Anomaly Labels')
plt.colorbar()
#plt.show()

# Scatter Plot with Anomaly Labels
plt.figure(figsize=(12, 8))
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 3], c=y_pred, cmap='coolwarm', s=50, alpha=0.7)
plt.xlabel(X_train.columns[0])  # Use the first feature name as x-axis label
plt.ylabel(X_train.columns[3])  # Use the second feature name as y-axis label
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
# plt.figure(figsize=(10, 6))
# plt.hist(data_without_nan['Total quantity_z_score'],bins=50,  label='Z-scores')
# plt.axvline(3, color='red', linestyle='dashed', linewidth=2, label='Outlier Threshold')
# plt.title('Histogram of Z-scores for Total Inventory')
# plt.xlabel('Z-score')
# plt.ylabel('Frequency')
# plt.legend()
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
# plt.figure(figsize=(10, 6))
# plt.hist(data_without_nan['Total quantity_z_score'],  label='Z-scores')
# plt.axvline(3, color='red', linestyle='dashed', linewidth=2, label='Outlier Threshold')
# plt.title('Histogram of Z-scores for Total Inventory')
# plt.xlabel('Z-score')
# plt.ylabel('Frequency')
# plt.legend()
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
# data['anomaly'] = data[[col + '_outlier' for col in numerical_columns]].max(axis=1)
#X = X_train

#X = data.drop(columns=['anomaly'])

# Data2=Data_Preprocessing(file_path=file_path)
# X_imputed= Data2.preprocess_data_kmean2()

#y = data['anomaly']


X_train, Y_train, x_val, Y_val = rfc_new.load_data(train_filepath, val_filepath)


# Train Isolation Forest model for anomaly detection
if_model = IsolationForest(random_state=42)
if_model.fit(X_train,Y_train)



Y_pred_train = if_model.predict(X_train)
Y_pred_val = if_model.predict(x_val)
Y_pred_val = np.where(Y_pred_val == -1, 1, 0)  # Convert -1 to 1 for anomaly, 1 to 0 for normal


print("Training Accuracy:", accuracy_score(Y_train, Y_pred_train))
print("Validation Accuracy:", accuracy_score(Y_val, Y_pred_val))
print("\nValidation Classification Report:\n", classification_report(Y_val, Y_pred_val))
print("\nValidation Confusion Matrix:\n", confusion_matrix(Y_val, Y_pred_val))


not_processed_data=pd.DataFrame(X_train)
# Predict anomaly labels using Isolation Forest
data2 = data.copy()
not_processed_data['if_anomaly'] = if_model.predict(X_train)
# Convert -1 to 1 for anomaly, 1 to 0 for normal
# Get anomaly scores for each sample
anomaly_scores = if_model.decision_function(X_train)

# Find the indices of the top 5 anomalies (lowest anomaly scores)
top_5_anomalies_indices = np.argsort(anomaly_scores)[:200]

# Get the corresponding rows (samples) from the original data
top_5_anomalies_data = not_processed_data.iloc[top_5_anomalies_indices]
# Save the top anomalies to a CSV file
top_5_anomalies_data.to_csv('top_150_Iforest_anomalies.csv', index=True)

#not_processed_data['if_anomaly'].to_csv('allanomaliIforest.csv', index=False)
# Find the top 5 columns with the most anomalies predicted by Isolation Forest
top_5_columns = np.argsort(np.sum(np.abs(if_model.decision_function(X_train))))

#Y_pred_val.iloc[top_5_anomalies_indices] = 1  # Convert anomalies to 1
#not_processed_data['if_anomaly'].fillna(0, inplace=True)  # Fill non-anomalies with 0

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you already have X_imputed and if_model from your previous code

# Get anomaly scores for each sample
anomaly_scores = if_model.decision_function(X_train)

# Custom estimator wrapper for IsolationForest
class IsolationForestWrapper(IsolationForest):
    def score(self, X, y=None):
        return np.mean(self.decision_function(X))

# Use the custom estimator wrapper
if_model_wrapped = IsolationForestWrapper(random_state=42)
if_model_wrapped.fit(X_train)

# Custom scoring function for anomaly detection
def custom_score(estimator, X, y):
    scores = estimator.decision_function(X)
    return np.mean(scores)

perm_importance = permutation_importance(if_model_wrapped, X_train, Y_train,scoring=custom_score, n_repeats=10, random_state=42)
# Perform permutation importance using the custom scoring function

# Create a DataFrame to display permutation importances
perm_importances_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
})
perm_importances_df.to_csv('feature_importances_If.csv')

import matplotlib.pyplot as plt

# Sort features by their mean importance score
sorted_indices = perm_importance.importances_mean.argsort()[::-1]

# Get sorted feature names and importances
sorted_feature_names = perm_importances_df['feature'].iloc[sorted_indices]
sorted_importances = perm_importances_df['importance'].iloc[sorted_indices]

# Plot the top N most important features
top_n = 8  # Adjust the number of top features you want to plot
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), sorted_importances[:top_n], align='center')
plt.yticks(range(top_n), sorted_feature_names[:top_n])
plt.xlabel('Permutation Importance')
plt.title('Top {} Most Important Features (Isolation Forest)'.format(top_n))
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
#plt.show()

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
correlation_matrix = X_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

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
    plt.show()

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








