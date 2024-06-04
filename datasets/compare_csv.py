import pandas as pd


xgbooost = pd.read_csv('anomalies/detected_anomalies_xgboost.csv')
iforest = pd.read_csv('anomalies/top_150_Iforest_anomalies.csv')
lof = pd.read_csv('anomalies/lof_with_anomalies.csv')
kmean= pd.read_csv('anomalies/anomalies_kmean.csv')
rfc= pd.read_csv('anomalies/rfc_anomaly_labels.csv')

rfc=rfc[rfc['Predicted_Label']==1]
lof=lof[lof['anomaly']==1]

print(lof)
print(kmean)
# Merge the dataframes on the index column
xgboost_iforest = pd.merge(xgbooost, iforest, on=xgbooost.columns[0])

xgboost_lof = pd.merge(xgbooost, lof, on=xgbooost.columns[0])

xgboost_kmean = pd.merge(xgbooost, kmean, on=xgbooost.columns[0])


xgboost_rfc = pd.merge(xgbooost, rfc, on=xgbooost.columns[0])


iforest_lof = pd.merge(iforest, lof, on=iforest.columns[0])

iforest_lof_xgboost_kmean= pd.merge(iforest_lof, xgboost_kmean, on=iforest.columns[0])



iforest_lof.to_csv(f'Intersections/anomalies_iforest_lof.csv', index=False)

xgboost_iforest.to_csv(f'Intersections/anomalies_xgboost_iforest.csv', index=False)

xgboost_lof.to_csv('Intersections/anomalies_xgboost_lof.csv', index=False)
xgboost_rfc.to_csv('Intersections/anomalies_xgboost_rfc.csv', index=False)

xgboost_kmean.to_csv('Intersections/anomalies_xgboost_kmean.csv', index=False)

iforest_lof_xgboost_kmean.to_csv('Intersections/anomalies_iforest_lof_xgboost_kmean.csv', index=False)



# Concatenate the dataframes
combined = pd.concat([xgbooost, iforest])

# Drop duplicate rows
combined_no_duplicates = combined.drop_duplicates()

# Save the combined file without duplicates
#combined_no_duplicates.to_csv('combined_no_duplicates.csv', index=False)