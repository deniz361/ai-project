import pandas as pd
import numpy as np



class Data_Preprocessing():
    def __init__(self, file_path) -> None:
        self.data = pd.read_csv(file_path)

    def preprocess_data(self):
        # Specify categorical and numerical numbers manually
        categorical_columns = ["Materialnummer", "Lieferant OB", "Vertragsposition OB", "Beschaffungsart", "Disponent", "Einkäufer", "Dispolosgröße", "Werk OB", "Warengruppe", "Basiseinheit"]
        numerical_columns = ["Planlieferzeit Vertrag", "Vertrag Fix1", "Vertrag Fix2", "Gesamtbestand", "Gesamtwert", "Preiseinheit", "WE-Bearbeitungszeit", "Planlieferzeit Mat-Stamm"]
        
        self.data[categorical_columns] = self.data[categorical_columns].astype('category')
        self.data[numerical_columns] = self.data[numerical_columns].astype('int64')

        # !!! The following is optional and can be commented out !!!:

        # Rename the columns to have a better understanding of what they actually mean
        self.data.nename(columns={"Lieferant OB": "Lieferant", "Vertragsposition OB": "Vertragsposition", 
                                  "Werk OB": "Werk", "Vertrag_Fix2": "Vertrag Fix2", "WE-Bearbeitungszeit": 
                                  "Bearbeitungszeit Werk"})

        # If Delivery time is 0, the value is missing
        self.data["Planlieferzeit Vertrag"] = self.data["Planlieferzeit Vertrag"].replace(0, np.nan)
        self.data["Planlieferzeit Mat-Stamm"] = self.data["Planlieferzeit Mat-Stamm"].replace(0, np.nan)

        # If processing time is 0, the value is missing
        self.data["Bearbeitungszeit Werk"] = self.data["Bearbeitungszeit Werk"].replace(0, np.nan)

        # If total quantity is 0, the value is missing
        self.data["Gesamtbestand"] = self.data["Gesamtbestand"].replace(0, np.nan)

        # If total value is 0, the toal value is not known or missing
        self.data["Gesamtwert"] = self.data["Gesamtwert"].replace(0, np.nan)


        return self.data
