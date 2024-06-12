import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_csv, train_dir, val_dir, test_dir, ratios=(0.85, 0.10, 0.05), seed=42):
    train_size, val_size, test_size = ratios
    assert train_size + val_size + test_size == 1, "Train, validation and test sizes must sum to 1."
    
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Split the dataset
    df_train, df_temp = train_test_split(df, train_size=train_size, random_state=seed)
    df_val, df_test = train_test_split(df_temp, test_size=test_size / (val_size + test_size), random_state=seed)
    
    # Ensure the output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Save the splits to CSV files
    df_train.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(val_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(test_dir, 'test.csv'), index=False)
    
    print("Datasets split and saved successfully.")
    
# Example usage:
input_csv = '/Users/awthura/THD/ai-project/datasets/supervised_dataset.csv'
train_dir = '/Users/awthura/THD/ai-project/datasets/split/supervised_learning_dataset/'
val_dir = '/Users/awthura/THD/ai-project/datasets/split/supervised_learning_dataset/'
test_dir = '/Users/awthura/THD/ai-project/datasets/split/supervised_learning_dataset/'

split_dataset(input_csv, train_dir, val_dir, test_dir)
