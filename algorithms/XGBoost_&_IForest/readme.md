### Preprocesinng 
For xgboost Iforest we used train.csv for the training. 

### Implementation 
In the unsupervised approach of XBoost and Iforest. We used the Z Score to find anomalies in the numeraical data.
When a numerical point is to 99% out of the mean value it is labeled as anomaly and trained with anomaly label.
The unsupervised were trained with labeled anomlies from train.csv, while the unsupervied were trained with Z Score of the numerical columns. 

### Takeways 
Iforest supervised cannot handle categorial data and make bad spuervised predictions.
XGBoost supervised makes good predictions but is probably overfitting
unsupervised Iforest and XGBoost cannot find any representiv anomalies with numerical Z Score outlier detection

### Anomaly Ratios
Here are the anomaly ratios for various categories, indicating the percentage of data points that significantly deviate from typical patterns:

- **Vertrag_Fix2_outlier**: 0.03
- **Preiseinheit_outlier**: 0.02
- **Vertrag Fix1_outlier**: 0.02
- **Planlieferzeit Vertrag_outlier**: 0.01
- **WE-Bearbeitungszeit_outlier**: 0.01
- **Gesamtwert_outlier**: 0.00
- **Gesamtbestand_outlier**: 0.00
- **Planlieferzeit Mat-Stamm_outlier**: 0.00

### Most Bizarre Values
Detailed explanations for the outliers which are the highest or most unusual values recorded in their respective categories:

- **Planlieferzeit Vertrag**: 392
  - **Description**: This value might occur due to being the highest value which is outside the normal range.

- **Vertrag Fix1**: 154
  - **Description**: This value might occur due to being the highest value which is outside the normal range.

- **Vertrag_Fix2**: 280
  - **Description**: This value might occur for the same reason as above.

- **Gesamtbestand**: 59,371,444
  - **Description**: This value might occur due to the value being too big; on average, the number lies within a 5-digit number.

- **Gesamtwert**: 115,344,675
  - **Description**: This value might occur due to the value being too big; on average, the number lies within a 7-digit number.

- **Preiseinheit**: 10,000
  - **Description**: This value might occur due to a typo mistake because the values are either 1 or 100, so it might be mistakenly typed.

- **WE-Bearbeitungszeit**: 56
  - **Description**: This value might occur due to being outside the normal range; the highest value in this category is 40.

- **Planlieferzeit Mat-Stamm**: 999
  - **Description**: This value is the most common, however, the algorithm has identified it as bizarre. Normally, the values lie in the range of 0 to 100, then a few within the 150 range, then most values are valued at 999.

