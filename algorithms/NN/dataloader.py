import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(filepath):
    # Read the CSV file
    df = pd.read_csv(filepath, low_memory=False)
    df = df.drop(columns=['Material number']).values

    # Identify categorical columns (Assuming these columns are known or identified programmatically)
    # For example, assuming columns 'cat1', 'cat2' are categorical
    categorical_columns = ["Supplier", "Contract Position", "Procurement type", "Dispatcher", "Buyer", "Purchasing lot size", "Plant", "Product group", "Base unit"]  # Modify these based on your dataset
    
    # Fill NaN values with the most frequent value in each column
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Fill NaN values in numerical columns with the most frequent value
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Standardize numerical columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Debug: Print the dataframe after preprocessing
    print("DataFrame after preprocessing:")
    print(df.head())
    print(f"Number of samples: {len(df)}")
    
    # Ensure the 'anomaly' column is present
    if 'anomaly' not in df.columns:
        raise ValueError("The column 'anomaly' does not exist in the dataset.")
    
    # Separate features and target
    X = df.drop(columns=['anomaly']).values
    Y = df['anomaly'].values
    
    # Check if there are enough samples
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("No valid samples found after processing the data.")
    
    # Split the data into training and testing sets (90:10 ratio)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    
    # Create DataLoader objects
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


