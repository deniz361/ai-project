import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

from dataloader import handle_zeros

def load_inference_data(filepath: str) -> DataLoader:
    # Identify categorical columns
    categorical_columns = ["Supplier", "Contract", "Contract Position", "Procurement type", "Dispatcher", "Buyer",
                           "Purchasing lot size", "Product group", "Base unit", 'Calendar', 'Purchasing group', 
                           'Special procurement type', 'Plant information record', 'Information record number', 
                           'Information record type']
    numeric_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value",
                       "Price unit", "Plant processing time", "Material master time"]

    # Load the label encoders
    with open('algorithms/NN/saved_prep_obj/label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

    # Load the categorical imputer
    with open('algorithms/NN/saved_prep_obj/cat_imputer.pkl', 'rb') as file:
        cat_imputer = pickle.load(file)

    # Load the numerical imputer
    with open('algorithms/NN/saved_prep_obj/num_imputer.pkl', 'rb') as file:
        num_imputer = pickle.load(file)

    # Load the scaler
    with open('algorithms/NN/saved_prep_obj/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Load data
    new_data = pd.read_csv(filepath)
    new_data = new_data.drop(columns=['Material number', 'Plant'])

    # Handle zeroes in new data
    new_data = handle_zeros(new_data)

    # Categorical preprocessing
    new_data[categorical_columns] = cat_imputer.transform(new_data[categorical_columns])
    for col in categorical_columns:
        le = label_encoders[col]
        # Add a special category for unseen labels
        new_data[col] = new_data[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')  # Add the special category to classes
        new_data[col] = le.transform(new_data[col])

    # Numerical preprocessing
    new_data[numeric_columns] = num_imputer.transform(new_data[numeric_columns])
    new_data[numeric_columns] = scaler.transform(new_data[numeric_columns])

    # Print the columns and shape before creating Tensor
    print(f"Columns after preprocessing: {new_data.columns.tolist()}")
    print(f"Shape after preprocessing: {new_data.shape}")

    # Drop the 'anomaly' column
    new_data = new_data.drop(columns=['anomaly'])

    # Create Tensor
    X_new_data = torch.tensor(new_data.values, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X_new_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    return loader

