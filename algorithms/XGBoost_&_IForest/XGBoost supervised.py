
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
x_val2=x_val
file_path = 'datasets/split/supervised_learning_dataset/train.csv'

Data=Data_Preprocessing(file_path=file_path)
data, not_processed_data, categorical_columns_encoded= Data.preprocess_data_kmean()

#data=pd.DataFrame(X_train,columns=numeric_columns)

Data=Data_Preprocessing(file_path=val_filepath)
val_data, not_processed_data, categorical_columns_encoded= Data.preprocess_data_kmean()

data=X_train





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


print(data['xgbPossibility'])
# Filter data for anomalies
# Filter not_processed_data for anomalies predicted by XGBoost
anomalies_xgboost = data[data['xgbPossibility'] > 0.995]




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


visualize_anomalies(data, y_pred)



# Find the top 5 columns with the most anomalies predicted by XGBoost
top_5_columns = xgb_model.feature_importances_.argsort()[-3:][::-1]

