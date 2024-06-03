import pandas as pd


xgbooost = pd.read_csv('detected_anomalies_xgboost.csv')
iforest = pd.read_csv('top_150_Iforest_anomalies.csv')
lof = pd.read_csv('processed_data_with_anomalies.csv')
dbscan= pd.read_csv('labeled_data.csv')

lof=lof[lof['anomaly']==1]
dbscan=dbscan[dbscan['anomaly']==1]
print(lof)
print(dbscan)
# Merge the dataframes on the index column
xgboost_iforest = pd.merge(xgbooost, iforest, on=xgbooost.columns[0])

xgboost_lof = pd.merge(xgbooost, lof, on=xgbooost.columns[0])

xgboost_dbscan = pd.merge(xgbooost, dbscan, on=xgbooost.columns[0])

iforest_lof = pd.merge(xgbooost, lof, on=iforest.columns[0])

iforest_lof_xgboost_dbscan= pd.merge(iforest_lof, xgboost_dbscan, on=iforest.columns[0])



iforest_lof.to_csv(f'Intersections/anomalies_iforest_lof.csv', index=False)

xgboost_iforest.to_csv(f'Intersections/anomalies_xgboost_iforest.csv', index=False)

xgboost_lof.to_csv('Intersections/anomalies_xgboost_lof.csv', index=False)

xgboost_dbscan.to_csv('Intersections/anomalies_xgboost_dbscan.csv', index=False)

iforest_lof_xgboost_dbscan.to_csv('Intersections/anomalies_iforest_lof_xgboost_dbscan.csv', index=False)



# Concatenate the dataframes
combined = pd.concat([xgbooost, iforest])

# Drop duplicate rows
combined_no_duplicates = combined.drop_duplicates()

# Save the combined file without duplicates
combined_no_duplicates.to_csv('combined_no_duplicates.csv', index=False)