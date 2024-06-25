
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
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
parent_dir2 = os.path.dirname(parent_dir)

sys.path.append(parent_dir2)
from datasets.data_preprocessing import Data_Preprocessing
from algorithms.RFC import xgb_preprocessing

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


X_train, Y_train, x_val, Y_val = xgb_preprocessing.load_data(train_filepath, val_filepath)
data=X_train


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








