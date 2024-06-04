
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
parent_dir2 = os.path.dirname(parent_dir)

sys.path.append(parent_dir2)

# Load the final_anomalies dataset (training data)
final_anomalies = pd.read_csv('datasets/final_anomalies.csv')

# Load the sampled dataset (test data)
sampled_data = pd.read_csv('datasets/sampled_data.csv', nrows=10000)



from datasets.data_preprocessing import Data_Preprocessing
file_path = 'datasets/sampled_data.csv'

Data=Data_Preprocessing(file_path=file_path)
data= Data.preprocess_data_kmean2()

# Check the distribution of the anomaly_normal column
label_distribution = final_anomalies['anomaly_normal'].value_counts()
print(label_distribution)



final_anomalies.info()


sampled_data.info()




# Prepare the training data
X_train = final_anomalies[Data.numerical_columns]
y_train = final_anomalies['anomaly_normal']


# Prepare the test data
X_test = sampled_data[Data.numerical_columns]


# Fill missing values (if any) with the mean of the respective column
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())


# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Evaluate on the training set
y_train_pred = rf_model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Classification Report:\n", classification_report(y_train, y_train_pred))


# Test on the sampled dataset
y_test_pred = rf_model.predict(X_test)
predictions = pd.DataFrame(y_test_pred, columns=['Predicted_Anomalies'])


# Get the anomaly scores (probabilities) for the test data
anomaly_scores = rf_model.predict_proba(X_test)[:, 0]  # Probability of being an anomaly (class 0)


# Add the anomaly scores as a new column to the test data
sampled_data['Anomaly_Score'] = anomaly_scores


# Save the DataFrame with anomaly scores to a CSV file
sampled_data.to_csv('sampled_data_with_anomaly_scores.csv', index=False)

# Print the first few rows of the DataFrame
print(sampled_data.head())


# Print the first few predictions
print(predictions.head())


from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())
print("Standard Deviation of Cross-Validation Scores:", cv_scores.std())



import matplotlib.pyplot as plt
import numpy as np
top_n=8
# Get feature importances from the trained model
top_features_indices = rf_model.feature_importances_.argsort()[-top_n:][::-1]
top_feature_names = X_train.columns[top_features_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features_indices)), rf_model.feature_importances_[top_features_indices], align='center')
plt.yticks(range(len(top_features_indices)), top_feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title(f'Top {top_n} Feature Importance Plot')
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, top_feature_names, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Model')
plt.show()


sampled_data = pd.read_csv('sampled_data_with_anomaly_scores.csv')


# Analyze the distribution of anomaly scores
plt.hist(sampled_data['Anomaly_Score'], bins=50, edgecolor='k')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Distribution of Anomaly Scores')
plt.show()


threshold = 0.5


# Label data based on the threshold
sampled_data['Predicted_Label'] = (sampled_data['Anomaly_Score'] < threshold).astype(int)

# Check the distribution of the predicted labels
label_distribution = sampled_data['Predicted_Label'].value_counts()
print(label_distribution)


# Save the labeled data to a new CSV file
sampled_data.to_csv('sampled_data_with_anomaly_labels.csv', index=True)

# Print the first few rows of the labeled data
print(sampled_data.head())





