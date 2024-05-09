import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

def sample_and_save_data(original_data, sample_fraction=0.1, random_state=42, file_path='sampled_data.csv'):
    """
    Sample the original data, save it to a CSV file, and return the sampled DataFrame.

    Parameters:
    - original_data: pandas DataFrame containing the original dataset
    - sample_fraction: Fraction of the original data to sample (default=0.01)
    - random_state: Random state for reproducibility (default=42)
    - file_path: File path to save the sampled data (default='sampled_data.csv')

    Returns:
    - sampled_data: pandas DataFrame containing the sampled dataset
    """
    sampled_data = original_data.sample(frac=sample_fraction, random_state=random_state)
    sampled_data.to_csv(file_path, index=False)
    return sampled_data

def load_data_and_run_lof(file_path, n_neighbors=10, contamination=0.1):
    """
    Load the sampled data from a CSV file, run the Local Outlier Factor (LOF) algorithm,
    and return anomalies detected along with the LOF scores.

    Parameters:
    - file_path: File path to load the sampled data from
    - n_neighbors: Number of neighbors to consider for LOF calculation (default=20)
    - contamination: Expected proportion of outliers in the dataset (default=0.1)

    Returns:
    - anomalies: pandas DataFrame containing the anomalies identified by LOF
    - lof_scores: LOF scores for each data point
    """
    # Load the sampled data
    sampled_data = pd.read_csv(file_path)

    # Preprocess the sampled data
    processed_sampled_data = preprocess_data(sampled_data)

    # Detect anomalies using LOF
    anomalies, lof_scores = detect_anomalies_LOF(processed_sampled_data, n_neighbors=n_neighbors, contamination=contamination)

    return anomalies, lof_scores

def preprocess_data(data):
    """
    Preprocess the data by imputing missing values, performing one-hot encoding for categorical variables,
    and performing feature normalization.

    Parameters:
    - data: pandas DataFrame containing the dataset

    Returns:
    - processed_data: pandas DataFrame containing imputed missing values, one-hot encoded features, and normalized features
    """
    # Separate numeric and categorical columns
    # numeric_cols = data.select_dtypes(include=np.number).columns
    # categorical_cols = data.select_dtypes(include='object').columns

    categorical_cols = ["Materialnummer", "Lieferant OB", "Vertragsposition OB", "Beschaffungsart", "Disponent", "Einkäufer", "Dispolosgröße", "Werk OB", "Warengruppe", "Basiseinheit"]
    numeric_cols = ["Planlieferzeit Vertrag", "Vertrag Fix1", "Vertrag_Fix2", "Gesamtbestand", "Gesamtwert", "Preiseinheit", "WE-Bearbeitungszeit", "Planlieferzeit Mat-Stamm"]
    
    # Impute missing values using mean imputation for numeric columns
    imputer = SimpleImputer(strategy='mean')
    data_numeric_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)

    # One-hot encode categorical variables
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first')
        data_encoded = encoder.fit_transform(data[categorical_cols])
        column_names = encoder.get_feature_names_out(categorical_cols)
        data_imputed_encoded = pd.DataFrame(data_encoded.toarray(), columns=column_names)
    else:
        data_imputed_encoded = pd.DataFrame()

    # Combine numeric and encoded categorical columns
    processed_data = pd.concat([data_numeric_imputed, data_imputed_encoded], axis=1)

    # Normalize features
    scaler = StandardScaler()
    processed_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns)

    return processed_data

def detect_anomalies_LOF(data, n_neighbors=20, contamination=0.1):
    """
    Detect anomalies using Local Outlier Factor (LOF) algorithm.

    Parameters:
    - data: pandas DataFrame containing the dataset
    - n_neighbors: Number of neighbors to consider for LOF calculation (default=20)
    - contamination: Expected proportion of outliers in the dataset (default=0.1)

    Returns:
    - anomalies: pandas DataFrame containing the anomalies identified by LOF
    - lof_scores: LOF scores for each data point
    """
    # Initialize LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    # Fit LOF to the data and predict anomaly scores
    lof_scores = lof.fit_predict(data)

    # Create a boolean mask to identify anomalies (outliers have a value of -1)
    anomalies_mask = lof_scores == -1

    # Filter anomalies
    anomalies = data[anomalies_mask]

    return anomalies, lof_scores

# Load the original dataset
file_path = 'ai-project/datasets/Stammdaten.csv'
original_data = pd.read_csv(file_path, low_memory=False)

# Sample and save the data
sampled_data = sample_and_save_data(original_data)

# Load the sampled data, run LOF, and detect anomalies
anomalies, lof_scores = load_data_and_run_lof('sampled_data.csv')

# Create a new column "anomaly" in the sampled data DataFrame
sampled_data['anomaly'] = np.where(lof_scores < 0, 1, 0)

# Save the sampled data with the "anomaly" column as CSV
sampled_data.to_csv('sampled_data_with_anomalies.csv', index=False)

print("Anomalies:")
print(anomalies)
print("\nLOF scores:")
print(lof_scores)

# Print indices of the sampled data DataFrame
print("Indices of sampled data:")
print(sampled_data.index)

# Print indices of the anomalies DataFrame
print("\nIndices of anomalies:")
print(anomalies.index)
