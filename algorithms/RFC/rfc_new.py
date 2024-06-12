import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle

def handle_zeros(data):
    columns_with_zero_as_missing = [
        "Fulfillment time", "Material master time", "Plant processing time",
        "Total quantity", "Total value", "Fixed contract 1", "Fixed contract 2"
    ]
    data[columns_with_zero_as_missing] = data[columns_with_zero_as_missing].replace(0, np.nan)
    return data

def load_data(train_filepath, val_filepath):
    train_df = pd.read_csv(train_filepath, low_memory=False)
    val_df = pd.read_csv(val_filepath, low_memory=False)

    train_df = train_df.drop(columns=['Material number', 'Plant'])
    val_df = val_df.drop(columns=['Material number', 'Plant'])

    print("Data types of training dataset:")
    print(train_df.dtypes)
    print("Data types of validation dataset:")
    print(val_df.dtypes)

    categorical_columns = ["Supplier", "Contract", "Contract Position", "Procurement type", "Dispatcher", "Buyer",
                           "Purchasing lot size", "Product group", "Base unit", 'Calendar', 'Purchasing group', 
                           'Special procurement type', 'Plant information record', 'Information record number', 
                           'Information record type']
    numeric_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value",
                       "Price unit", "Plant processing time", "Material master time"]

    train_df = handle_zeros(train_df)
    val_df = handle_zeros(val_df)

    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    imputer = SimpleImputer(strategy='most_frequent')
    combined_df[categorical_columns] = imputer.fit_transform(combined_df[categorical_columns])

    with open('algorithms/RFC/saved_prep_obj/cat_imputer.pkl', 'wb') as file:
        pickle.dump(imputer, file)

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
        label_encoders[col] = le

    with open('algorithms/RFC/saved_prep_obj/label_encoders.pkl', 'wb') as file:
        pickle.dump(label_encoders, file)

    combined_df[numeric_columns] = imputer.fit_transform(combined_df[numeric_columns])

    with open('algorithms/RFC/saved_prep_obj/num_imputer.pkl', 'wb') as file:
        pickle.dump(imputer, file)

    scaler = StandardScaler()
    combined_df[numeric_columns] = scaler.fit_transform(combined_df[numeric_columns])

    with open('algorithms/RFC/saved_prep_obj/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    print("Columns with NaN values after preprocessing:")
    print(combined_df.columns[combined_df.isna().any()].tolist())

    print("Combined DataFrame after preprocessing:")
    print(combined_df.head())
    print(f"Number of samples: {len(combined_df)}")

    train_size = len(train_df)
    train_df = combined_df.iloc[:train_size]
    val_df = combined_df.iloc[train_size:]

    if 'anomaly' not in train_df.columns or 'anomaly' not in val_df.columns:
        raise ValueError("The column 'anomaly' does not exist in the dataset.")

    X_train = train_df.drop(columns=['anomaly']).values
    Y_train = train_df['anomaly'].values
    X_val = val_df.drop(columns=['anomaly']).values
    Y_val = val_df['anomaly'].values

    return X_train, Y_train, X_val, Y_val

def train_random_forest(X_train, Y_train, X_val, Y_val):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, Y_train)

    Y_pred_train = rf_classifier.predict(X_train)
    Y_pred_val = rf_classifier.predict(X_val)

    print("Training Accuracy:", accuracy_score(Y_train, Y_pred_train))
    print("Validation Accuracy:", accuracy_score(Y_val, Y_pred_val))
    print("\nValidation Classification Report:\n", classification_report(Y_val, Y_pred_val))
    print("\nValidation Confusion Matrix:\n", confusion_matrix(Y_val, Y_pred_val))

    with open('algorithms/RFC/saved_prep_obj/random_forest_model.pkl', 'wb') as file:
        pickle.dump(rf_classifier, file)

if __name__ == "__main__":
    train_filepath = 'datasets/split/supervised_learning_dataset/train.csv'
    val_filepath = 'datasets/split/supervised_learning_dataset/val.csv'
    
    X_train, Y_train, X_val, Y_val = load_data(train_filepath, val_filepath)
    train_random_forest(X_train, Y_train, X_val, Y_val)

