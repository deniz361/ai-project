
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load the final_anomalies dataset (training data)
final_anomalies = pd.read_csv('datasets/final_anomalies.csv')

# Load the sampled dataset (test data)
sampled_data = pd.read_csv('datasets/sampled_data.csv', nrows=10000)


# Check the distribution of the anomaly_normal column
label_distribution = final_anomalies['anomaly_normal'].value_counts()
print(label_distribution)



final_anomalies.info()


sampled_data.info()


# Define the features to be used
features = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value", "Price unit", "Plant processing time", "Material master time"]
# Rename the column in the test data to match the training data
sampled_data.rename(columns={'WE-Bearbeitungszeit': 'WE_Bearbeitungszeit'}, inplace=True)
features = ["Planlieferzeit Vertrag", "Vertrag Fix1", "Vertrag_Fix2", "Gesamtbestand", "Gesamtwert", "Preiseinheit", "WE_Bearbeitungszeit", "Planlieferzeit Mat-Stamm"]


# Prepare the training data
X_train = final_anomalies[features]
y_train = final_anomalies['anomaly_normal']
y_train = y_train.rename(columns={"Planlieferzeit Vertrag":"Fulfillment time", "Vertrag Fix1":"Fixed contract 1", "Vertrag_Fix2":"Fixed contract 2", "Gesamtbestand":"Total quantity", "Gesamtwert":"Total value", "Preiseinheit":"Price unit", "WE_Bearbeitungszeit":"Plant processing time", "Planlieferzeit Mat-Stamm":"Material master time"}, inplace=True)


# Prepare the test data
X_test = sampled_data[features]


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

# Get feature importances from the trained model
feature_importances = rf_model.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue')
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
sampled_data.to_csv('sampled_data_with_anomaly_labels.csv', index=False)

# Print the first few rows of the labeled data
print(sampled_data.head())





