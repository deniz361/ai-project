
from turtle import pd
import pandas as pd
import sys
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
parent_dir2 = os.path.dirname(parent_dir)

sys.path.append(parent_dir2)



from datasets.data_preprocessing import Data_Preprocessing
sys.path.append('../')  # Add the parent directory to the Python path

import numpy as np
from sklearn.cluster import KMeans

def preprocess_data(data):
    # Drop columns with NaN values

    categorical_columns = ["Materialnummer", "Lieferant OB", "Vertragsposition OB", "Beschaffungsart", "Disponent", "Einkäufer", "Dispolosgröße", "Werk OB", "Warengruppe", "Basiseinheit"]
    numerical_columns = ["Planlieferzeit Vertrag", "Gesamtbestand", "Gesamtwert", "Preiseinheit", "WE-Bearbeitungszeit", "Planlieferzeit Mat-Stamm"]

    data[categorical_columns] = data[categorical_columns].astype('category')
    data[numerical_columns] = data[numerical_columns].astype('int64')

    # Select only numeric columns
    processed_data = data[numerical_columns]
    processed_data = processed_data.dropna(axis=1)

    # Normalize features
    #scaler = StandardScaler()
    #processed_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns)

    return processed_data  

file_path = 'datasets/sampled_data.csv'
df = pd.read_csv(file_path, low_memory=False, nrows=10000)

Data=Data_Preprocessing(file_path=file_path)
processed_data= Data.preprocess_data_kmean2()


print(processed_data)


def detect_anomalies(data, num_clusters, threshold):
    # Fit KMeans clustering model
    kmeans = KMeans(n_clusters=num_clusters)

    kmeans.fit(data)
    
    # Calculate distances of each point to its nearest cluster center
    distances = kmeans.transform(data)
    
    # Calculate average distance to cluster centers for each point
    avg_distances = np.mean(distances, axis=1)
    
    # Calculate threshold for anomaly detection
    anomaly_threshold = np.percentile(avg_distances, threshold)
    
    # Identify anomalies
    anomalies = data[avg_distances > anomaly_threshold]
    
    return anomalies
# dropna does not work no data
# processed_data=processed_data.dropna()

anomalies = detect_anomalies(processed_data, num_clusters=5, threshold=98)
#unprocessed_anomalies = not_scaled_data.loc[anomalies.index]
print(anomalies)

anomalies.to_csv('algorithms/Kmeans/anomalies_kmean.csv', index=True)

