import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class Data_Preprocessing():
    def __init__(self, file_path) -> None:
        self.data = pd.read_csv(file_path, low_memory=False)
        
        self.rename_to_english()
        
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
    
