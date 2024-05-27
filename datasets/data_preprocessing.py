import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



class Data_Preprocessing():
    def __init__(self, file_path) -> None:
        self.data = pd.read_csv(file_path, low_memory=False, nrows=10000)
        
        self.rename_to_english()

        self.data.reset_index()
        
        print(self.data.columns)
        
        # Specify categorical and numerical numbers manually
        self.categorical_columns = ["Material number", "Supplier", "Contract", "Contract Position", "Procurement type", 
                                    "Special procurement type", "Dispatcher", "Buyer", "Purchasing group", 
                                    "Purchasing lot size", "Calendar", "Plant", "Plant information record", 
                                    "Information record number", "Information record type",  "Product group",
                                    "Base unit"]
        self.numerical_columns = ["Fulfillment time", "Fixed contract 1", "Fixed contract 2", "Total quantity", "Total value", 
                                  "Price unit", "Plant processing time", "Material master time"]
        
        # Initialize MinMaxScaler
        self.scaler = MinMaxScaler()
    
    def rename_to_english(self):
        self.data.rename(columns={"Materialnummer": "Material number", "Lieferant OB": "Supplier", "Vertrag OB": "Contract", 
                                 "Vertragsposition OB": "Contract Position", "Planlieferzeit Vertrag": "Fulfillment time", 
                                 "Vertrag Fix1": "Fixed contract 1", "Vertrag_Fix2": "Fixed contract 2", "Beschaffungsart": 
                                 "Procurement type", "Sonderbeschaffungsart": "Special procurement type", "Disponent":
                                 "Dispatcher", "Einkäufer": "Buyer", "DispoGruppe": "Purchasing group", "Dispolosgröße": 
                                 "Purchasing lot size", "Gesamtbestand": "Total quantity", "Gesamtwert": "Total value",
                                 "Preiseinheit": "Price unit", "Kalender": "Calendar", "Werk OB": "Plant", "Werk Infosatz":
                                 "Plant information record", "Infosatznummer": "Information record number", "Infosatztyp":
                                 "Information record type", "WE-Bearbeitungszeit": "Plant processing time", "Planlieferzeit Mat-Stamm":
                                 "Material master time", "Warengruppe": "Product group", "Basiseinheit": "Base unit"}, inplace=True)

    def normalize_data(self, numerical_data):
        # Fit and transform the data
        normalized_data = self.scaler.fit_transform(numerical_data)

        return normalized_data

    def preprocess_data(self):
        self.data[self.categorical_columns] = self.data[self.categorical_columns].astype('category')
        self.data[self.numerical_columns] = self.data[self.numerical_columns].astype('int64')


        # If Delivery time is 0, the value is missing
        self.data["Fulfillment time"] = self.data["Fulfillment time"].replace(0, np.nan)
        self.data["Material master time"] = self.data["Material master time"].replace(0, np.nan)

        # If processing time is 0, the value is missing
        self.data["Plant processing time"] = self.data["Plant processing time"].replace(0, np.nan)

        # If total quantity is 0, the value is missing
        self.data["Total quantity"] = self.data["Total quantity"].replace(0, np.nan)

        # If total value is 0, the toal value is not known or missing
        self.data["Total value"] = self.data["Total value"].replace(0, np.nan)

        self.data["Fixed contract 1"] = self.data["Fixed contract 1"].replace(0, np.nan)
        self.data["Fixed contract 2"] = self.data["Fixed contract 2"].replace(0, np.nan)


        return self.data
    

    def preprocess_dbscan(self, data):
        numerical_columns = ["Fulfillment time", "Fixed contract 1"]
        data = data[numerical_columns]

        # Remove rows with NaN values
        data_without_nan = data.dropna(axis=0)

        return data_without_nan
    
    def preprocess_data_kmean(self):
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

        #data = data[numerical_columns]

        self.data=self.preprocess_data()
        from sklearn.impute import KNNImputer

        # Instantiate KNN imputer
        imputer = KNNImputer()

        # Impute missing values using KNN imputation
        data_numeric_imputed = imputer.fit_transform(self.data[self.numerical_columns])
        data_numeric_imputed= pd.DataFrame(data_numeric_imputed,columns=self.numerical_columns)
        not_scaled_data = self.data.copy()

        # categorical_cols = ["Materialnummer", "Lieferant OB", "Vertragsposition OB", "Beschaffungsart", "Disponent", "Einkäufer", "Dispolosgröße", "Werk OB", "Warengruppe", "Basiseinheit"]
        # numeric_cols = ["Planlieferzeit Vertrag", "Vertrag Fix1", "Vertrag_Fix2", "Gesamtbestand", "Gesamtwert", "Preiseinheit", "WE-Bearbeitungszeit", "Planlieferzeit Mat-Stamm"]
        
        # Impute missing values using mean imputation for numeric columns
        # imputer = SimpleImputer(strategy='mean')
        # data_numeric_imputed = pd.DataFrame(imputer.fit_transform(self.data[self.numerical_columns]), columns=self.numerical_columns)

        # One-hot encode categorical variables
        if len(self.categorical_columns) > 0:
            encoder = OneHotEncoder(drop='first')
            data_encoded = encoder.fit_transform(self.data[self.categorical_columns])
            column_names = encoder.get_feature_names_out(self.categorical_columns)
            data_imputed_encoded = pd.DataFrame(data_encoded.toarray(), columns=column_names)
        else:
            data_imputed_encoded = pd.DataFrame()

        # Combine numeric and encoded categorical columns
        processed_data = pd.concat([data_numeric_imputed, data_imputed_encoded], axis=1)


        # Normalize features
        scaler = StandardScaler()
        processed_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns)


        return processed_data, not_scaled_data
