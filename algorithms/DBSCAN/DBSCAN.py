import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from DBSCANAnalyzer import DBSCANAnalyzer

def detect_anomalies_DBSCAN(data, eps=1.90, min_samples=2):
    """
    Detect anomalies using DBSCAN algorithm.

    Parameters:
    - data: pandas DataFrame containing the dataset
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. (default=0.5)
    - min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. (default=5)

    Returns:
    - anomalies: pandas DataFrame containing the anomalies identified by DBSCAN
    """
    # Initialize DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit DBSCAN to the data
    dbscan.fit(data)

    # Get labels assigned to each point by DBSCAN (-1 indicates anomaly)
    labels = dbscan.labels_

    # Create a new DataFrame with an additional "anomaly" column
    labeled_data = data.copy()
    labeled_data['anomaly'] = np.where(labels == -1, 1, 0)
    
    # Filter anomalies
    anomalies_mask = labels == -1
    anomalies = data[anomalies_mask]

    return anomalies, labels, labeled_data

def print_unique_values_by_dtype(data):
    """
    Print the number of unique numeric and non-numeric values for each column in the DataFrame.
    Print columns containing NaN values and count the number of NaN values for each column.

    Parameters:
    - data: pandas DataFrame containing the dataset
    """
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            unique_values = data[column].nunique()
            print(f"Numeric column '{column}': {unique_values} unique values")
        else:
            unique_values = data[column].nunique()
            print(f"Non-numeric column '{column}': {unique_values} unique values")
            
    columns_with_missing_values = data.columns[data.isnull().any()]
    
    if len(columns_with_missing_values) == 0:
        print("No columns contain NaN values.")
    else:
        print("Columns containing NaN values:")
        for column in columns_with_missing_values:
            count_missing_values = data[column].isnull().sum()
            print(f"Column '{column}': {count_missing_values} missing values")


def preprocess_data(data):
    """
    Preprocess the data by removing non-numeric columns, dropping columns with NaN values,
    imputing missing values, and performing feature normalization.

    Parameters:
    - data: pandas DataFrame containing the dataset

    Returns:
    - processed_data: pandas DataFrame containing only numeric columns, imputed missing values, and normalized features
    """
    # Drop columns with NaN values
    data = data.dropna(axis=1)

    categorical_columns = ["Materialnummer", "Lieferant OB", "Vertragsposition OB", "Beschaffungsart", "Disponent", "Einkäufer", "Dispolosgröße", "Werk OB", "Warengruppe", "Basiseinheit"]
    numerical_columns = ["Planlieferzeit Vertrag", "Vertrag Fix1", "Vertrag_Fix2", "Gesamtbestand", "Gesamtwert", "Preiseinheit", "WE-Bearbeitungszeit", "Planlieferzeit Mat-Stamm"]

    data[categorical_columns] = data[categorical_columns].astype('category')
    data[numerical_columns] = data[numerical_columns].astype('int64')

    # Select only numeric columns
    processed_data = data[numerical_columns]

    # Normalize features
    #scaler = StandardScaler()
    #processed_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns)

    return processed_data           
            
file_path = 'datasets/sampled_data.csv'
df = pd.read_csv(file_path, low_memory=False, nrows=1000)
processed_df = preprocess_data(df)
# print_unique_values_by_dtype(df) 
anomalies, labels, labeled_df = detect_anomalies_DBSCAN(processed_df)

dbscan_visualizer = DBSCANAnalyzer(processed_df, labels)
dbscan_visualizer.silhouette_score()

# Save the labeled DataFrame as a CSV file
labeled_df.to_csv('labeled_data.csv', index=True)