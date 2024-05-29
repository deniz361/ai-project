import pandas as pd

def rename(data):
    data.rename(columns={"Materialnummer": "Material number", "Lieferant OB": "Supplier", "Vertrag OB": "Contract", 
                         "Vertragsposition OB": "Contract Position", "Planlieferzeit Vertrag": "Fulfillment time", 
                         "Vertrag Fix1": "Fixed contract 1", "Vertrag_Fix2": "Fixed contract 2", "Beschaffungsart": 
                         "Procurement type", "Sonderbeschaffungsart": "Special procurement type", "Disponent":
                         "Dispatcher", "Einkäufer": "Buyer", "DispoGruppe": "Purchasing group", "Dispolosgröße": 
                         "Purchasing lot size", "Gesamtbestand": "Total quantity", "Gesamtwert": "Total value",
                         "Preiseinheit": "Price unit", "Kalender": "Calendar", "Werk OB": "Plant", "Werk Infosatz":
                         "Plant information record", "Infosatznummer": "Information record number", "Infosatztyp":
                         "Information record type", "WE-Bearbeitungszeit": "Plant processing time", "Planlieferzeit Mat-Stamm":
                         "Material master time", "Warengruppe": "Product group", "Basiseinheit": "Base unit"}, inplace=True)
    return data

def create_supervised_dataset(file1, file2, file3):
    # Read the CSV files into DataFrames with low_memory=False to handle mixed types
    df1 = pd.read_csv(file1, low_memory=False)
    df2 = pd.read_csv(file2, low_memory=False)
    df3 = pd.read_csv(file3, low_memory=False)

    # Rename columns in df1, df2, and df3
    df1 = rename(df1)
    df2 = rename(df2)
    df3 = rename(df3)

    # Ensure 'Material number' and 'plant id' columns exist in all DataFrames
    required_columns = {'Material number', 'Plant'}
    if not required_columns.issubset(df1.columns) or not required_columns.issubset(df2.columns) or not required_columns.issubset(df3.columns):
        raise ValueError("All datasets must contain the 'Material number' and 'Plant' columns.")

    # Create sets of ('Material number', 'plant id') tuples from df2 and df3
    material_plant_set_df2 = set(df2[['Material number', 'Plant']].apply(tuple, axis=1))
    material_plant_set_df3 = set(df3[['Material number', 'Plant']].apply(tuple, axis=1))

    # Union of both sets
    combined_material_plant_set = material_plant_set_df2.union(material_plant_set_df3)

    # Add 'anomaly' column to df1 and count matches
    df1['anomaly'] = df1.apply(lambda row: 1 if (row['Material number'], row['Plant']) in combined_material_plant_set else 0, axis=1)
    match_count = df1['anomaly'].sum()

    # The supervised dataset includes all columns from df1 plus the 'anomaly' column
    supervised_dataset = df1.copy()

    # Print the number of matches found
    print(f"Number of matches found for 'Material number' and 'Plant': {match_count}")

    return supervised_dataset

# Path to CSV files
file1 = '/Users/awthura/THD/ai-project/datasets/Stammdaten.csv'
file2 = '/Users/awthura/THD/ai-project/datasets/final_anomalies.csv'
file3 = '/Users/awthura/THD/ai-project/datasets/rule_based_anomalies.csv'

# Create the supervised dataset
supervised_dataset = create_supervised_dataset(file1, file2, file3)

# Save the supervised dataset to a new CSV file
supervised_dataset.to_csv('/Users/awthura/THD/ai-project/datasets/supervised_dataset.csv', index=False)

print("Supervised dataset created and saved to 'supervised_dataset.csv'.")
print(len(supervised_dataset))