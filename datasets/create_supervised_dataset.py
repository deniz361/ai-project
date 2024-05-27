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

    # Rename columns in df1
    df1 = rename(df1)
    df2 = rename(df2)

    # Ensure 'Material number' column exists in both DataFrames
    if 'Material number' not in df1.columns or 'Material number' not in df2.columns:
        raise ValueError("Both datasets must contain the 'Material number' column.")

    # Create a set of 'Material number' from df2
    material_numbers_df2 = set(df2['Material number'])

    # Add 'anomaly' column to df1 and count matches
    df1['anomaly'] = df1['Material number'].apply(lambda x: 1 if x in material_numbers_df2 else 0)
    match_count = df1['anomaly'].sum()

    # The supervised dataset includes all columns from df1 plus the 'anomaly' column
    supervised_dataset = df1.copy()

    # Print the number of matches found
    print(f"Number of matches found for 'Material number': {match_count}")

    return supervised_dataset

# Path to CSV files
file1 = '/Users/awthura/THD/ai-project/datasets/Stammdaten.csv'
file2 = '/Users/awthura/THD/ai-project/datasets/final_correct.csv'

# Create the supervised dataset
supervised_dataset = create_supervised_dataset(file1, file2)

# Save the supervised dataset to a new CSV file
supervised_dataset.to_csv('/Users/awthura/THD/ai-project/datasets/supervised_dataset.csv', index=False)

print("Supervised dataset created and saved to 'supervised_dataset.csv'.")
