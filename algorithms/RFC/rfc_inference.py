import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

def handle_zeros(data):
    """
    Replace zero values with NaN in specified columns.

    Args:
        data (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.DataFrame: DataFrame with zeros replaced by NaN in specified columns.
    """
    columns_with_zero_as_missing = [
        "Fulfillment time", "Material master time", "Plant processing time",
        "Total quantity", "Total value", "Fixed contract 1", "Fixed contract 2"
    ]
    data[columns_with_zero_as_missing] = data[columns_with_zero_as_missing].replace(0, np.nan)
    return data

def load_inference_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the data for inference.

    Args:
        filepath (str): Path to the inference data CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for inference.
    """
    categorical_columns = ["Supplier", "Contract", "Contract Position", "Procurement type", "Dispatcher", "Buyer",
                           "Purchasing lot size", "Product group", "Base unit", 'Calendar', 'Purchasing group', 
                           'Special procurement type', 'Plant information record', 'Information record number', 
                           'Information record type']
    numeric_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value",
                       "Price unit", "Plant processing time", "Material master time"]

    with open('algorithms/RFC/saved_prep_obj/label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

    with open('algorithms/RFC/saved_prep_obj/cat_imputer.pkl', 'rb') as file:
        cat_imputer = pickle.load(file)

    with open('algorithms/RFC/saved_prep_obj/num_imputer.pkl', 'rb') as file:
        num_imputer = pickle.load(file)

    with open('algorithms/RFC/saved_prep_obj/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    new_data = pd.read_csv(filepath)
    new_data = new_data.drop(columns=['Material number', 'Plant'])

    new_data = handle_zeros(new_data)

    new_data[categorical_columns] = cat_imputer.transform(new_data[categorical_columns])
    for col in categorical_columns:
        le = label_encoders[col]
        new_data[col] = new_data[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        new_data[col] = le.transform(new_data[col])

    new_data[numeric_columns] = num_imputer.transform(new_data[numeric_columns])
    new_data[numeric_columns] = scaler.transform(new_data[numeric_columns])

    print(f"Columns after preprocessing: {new_data.columns.tolist()}")
    print(f"Shape after preprocessing: {new_data.shape}")

    if 'anomaly' in new_data.columns:
        new_data = new_data.drop(columns=['anomaly'])

    return new_data

def predict_with_random_forest(filepath: str, model_path: str):
    """
    Load the trained RFC model and make predictions on new data.

    Args:
        filepath (str): Path to the inference data CSV file.
        model_path (str): Path to the saved RFC model file.
    """
    with open(model_path, 'rb') as file:
        rf_classifier = pickle.load(file)

    inference_data = load_inference_data(filepath)

    predictions = rf_classifier.predict(inference_data)

    original_data = pd.read_csv(filepath)
    original_data = original_data.drop(columns=['Material number', 'Plant'])
    original_data['Prediction'] = predictions

    original_data.to_csv('algorithms/RFC/results/predictions.csv', index=False)

    anomalies = original_data[original_data['Prediction'] == 1]
    anomalies.to_csv('algorithms/RFC/results/anomalies.csv', index=False)

    print("Predictions saved to predictions.csv")
    print("Anomalies saved to anomalies.csv")

if __name__ == "__main__":
    predict_with_random_forest("datasets/supervised_dataset.csv", "algorithms/RFC/saved_prep_obj/random_forest_model.pkl")
