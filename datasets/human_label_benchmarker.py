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

def create_supervised_dataset(file1, file2):
    # Read the CSV files into DataFrames with low_memory=False to handle mixed types
    df1 = pd.read_csv(file1, low_memory=False)
    df2 = pd.read_csv(file2, low_memory=False)

    # Rename columns in df1 and df2
    df1 = rename(df1)
    df2 = rename(df2)

    # Ensure 'Material number' and 'Plant' columns exist in both DataFrames
    required_columns = {'Material number', 'Plant'}
    if not required_columns.issubset(df1.columns) or not required_columns.issubset(df2.columns):
        raise ValueError("Both datasets must contain the 'Material number' and 'Plant' columns.")

    # Create a set of ('Material number', 'Plant') tuples from df2
    material_plant_set_df2 = set(df2[['Material number', 'Plant']].apply(tuple, axis=1))

    # Add 'anomaly' column to df1 based on the presence in df2
    df1['anomaly'] = df1.apply(lambda row: 1 if (row['Material number'], row['Plant']) in material_plant_set_df2 else 0, axis=1)
    match_count = df1['anomaly'].sum()

    # Keep only the rows where 'anomaly' is 1
    supervised_dataset = df1[df1['anomaly'] == 1]

    # Print the number of matches found
    print(f"Number of anomalies found: {match_count}")

    return supervised_dataset

# Path to the CSV files
file1 = '/Users/awthura/THD/ai-project/datasets/Stammdaten.csv'
file2 = '/Users/awthura/THD/ai-project/datasets/final_anomalies.csv'  # Merged anomaly file from final_anomalies and rule_based_anomalies

# Create the supervised dataset
supervised_dataset = create_supervised_dataset(file1, file2)

# Save the supervised dataset to a new CSV file
supervised_dataset.to_csv('/Users/awthura/THD/ai-project/datasets/human_labelled_anomalies.csv', index=False)

print("Supervised dataset created and saved to 'human_labelled_anomalies.csv'.")
print(len(supervised_dataset))
