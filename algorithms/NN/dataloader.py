import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def handle_zeros(data):

    # Replace 0 with NaN for specified columns
    columns_with_zero_as_missing = [
        "Fulfillment time", "Material master time", "Plant processing time",
        "Total quantity", "Total value", "Fixed contract 1", "Fixed contract 2"
    ]
    data[columns_with_zero_as_missing] = data[columns_with_zero_as_missing].replace(0, np.nan)

    return data

def load_data(train_filepath, val_filepath):
    # Read the CSV files
    train_df = pd.read_csv(train_filepath, low_memory=False)
    val_df = pd.read_csv(val_filepath, low_memory=False)
    
    # Drop unnecessary columns
    train_df = train_df.drop(columns=['Material number', 'Plant'])
    val_df = val_df.drop(columns=['Material number', 'Plant'])
    
    # Print data types of entire dataset
    print("Data types of training dataset:")
    print(train_df.dtypes)
    print("Data types of validation dataset:")
    print(val_df.dtypes)
    
    # Identify categorical columns
    categorical_columns = ["Supplier", "Contract", "Contract Position", "Procurement type", "Dispatcher", "Buyer", "Purchasing lot size",
                            "Product group", "Base unit" , 'Calendar', 'Purchasing group', 'Special procurement type' , 'Plant information record',
                             'Information record number', 'Information record type' ]
    numeric_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value", "Price unit", "Plant processing time", "Material master time"] 
    
    # Zero handling
    train_df = handle_zeros(train_df)
    val_df = handle_zeros(val_df)

    # Combine train and validation datasets temporarily
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Fill NaN values with the most frequent value in each column
    imputer = SimpleImputer(strategy='most_frequent')
    combined_df[categorical_columns] = imputer.fit_transform(combined_df[categorical_columns])
    
    # Label encode categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
        label_encoders[col] = le

    # Fill NaN values in numerical columns with the most frequent value
    combined_df[numeric_columns] = imputer.fit_transform(combined_df[numeric_columns])
    
    # Standardize numerical columns
    scaler = StandardScaler()
    combined_df[numeric_columns] = scaler.fit_transform(combined_df[numeric_columns])

    # Print columns with NaN values after preprocessing
    print("Columns with NaN values after preprocessing:")
    print(combined_df.columns[combined_df.isna().any()].tolist())

    # Debug: Print the dataframe after preprocessing
    print("Combined DataFrame after preprocessing:")
    print(combined_df.head())
    print(f"Number of samples: {len(combined_df)}")
    
    # Separate combined dataset into train and validation sets
    train_size = len(train_df)
    train_df = combined_df.iloc[:train_size]
    val_df = combined_df.iloc[train_size:]
    
    # Ensure the 'anomaly' column is present
    if 'anomaly' not in train_df.columns or 'anomaly' not in val_df.columns:
        raise ValueError("The column 'anomaly' does not exist in the dataset.")
    
    # Separate features and target for train and validation sets
    X_train = train_df.drop(columns=['anomaly']).values
    Y_train = train_df['anomaly'].values
    X_val = val_df.drop(columns=['anomaly']).values
    Y_val = val_df['anomaly'].values
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
    
    # Create DataLoader objects
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


