import torch
import pickle
import pandas as pd

from dataloader import handle_zeros

# Identify categorical columns
categorical_columns = ["Supplier", "Contract", "Contract Position", "Procurement type", "Dispatcher", "Buyer",
                       "Purchasing lot size",
                       "Product group", "Base unit", 'Calendar', 'Purchasing group', 'Special procurement type',
                       'Plant information record',
                       'Information record number', 'Information record type']
numeric_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value",
                   "Price unit", "Plant processing time", "Material master time"]


def main(filepath: str) -> torch.Tensor:
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
        new_data[col] = le.transform(new_data[col])

    # Numerical preprocessing
    new_data[numeric_columns] = num_imputer.transform(new_data[numeric_columns])
    new_data[numeric_columns] = scaler.transform(new_data[numeric_columns])

    # Create Tensor
    new_data = new_data.drop(columns=['anomaly'])  # optional (used for testing)
    return torch.tensor(new_data.values, dtype=torch.float32)


if __name__ == '__main__':
    print(main("datasets/split/supervised_learning_dataset/val.csv"))
