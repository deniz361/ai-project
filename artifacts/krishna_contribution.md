# My Contribution to the Group Project: Anomaly Detection and Data Transformation

In our group project, I focused on implementing various anomaly detection techniques, transforming the dataset for better analysis, and enhancing the performance of our models. My contributions are detailed below:

## 1. Data Preprocessing

### Initial Steps
- **Loading the Dataset**: Loaded datasets from CSV files, including training, testing, and validation sets.
- **Handling Missing Values**: Replaced missing values in numeric columns with the median to ensure data integrity.

### Data Transformation
- **Identifying Column Types**: Identified categorical and numerical columns for appropriate processing.
- **Encoding and Scaling**: Applied label encoding to categorical columns and standardized numerical columns using `StandardScaler`.

## 2. Anomaly Detection Techniques

### Z-score Method
- **Z-score Calculation**: Computed Z-scores for numerical columns to identify outliers.
- **Anomaly Labeling**: Flagged values with a Z-score greater than 3 as outliers and created a combined `anomaly_label`.

### XGBoost
- **Training**: Trained an XGBoost classifier on the numerical columns to predict anomaly labels.
- **Prediction**: Added model predictions as a new column, `xgb_anomaly`.

### Isolation Forest
- **Training**: Applied Isolation Forest on the numerical columns to detect anomalies.
- **Scoring and Visualization**: Computed anomaly scores and flags, and visualized these using histograms and scatter plots.

## 3. Dimensionality Reduction and Visualization

### Applying TruncatedSVD
- **Dimensionality Reduction**: Applied TruncatedSVD to reduce the processed dataset to a lower dimensionality while retaining significant variance.
- **Visualization**: Created scatter plot matrices for pairs of SVD components to identify patterns and relationships.

### Visualizing Anomaly Scores
- **Histogram of Z-scores**: Plotted histograms to visualize Z-scores.
- **Scatter Plots**: Displayed scatter plots of PCA-reduced datasets colored by anomaly scores.

## 4. Autoencoder and XGBoost for Enhanced Anomaly Detection

### Autoencoder for Feature Extraction
- **Building and Training**: Designed and trained an autoencoder to compress and encode categorical features.
- **Encoding Features**: Used the trained encoder to transform categorical features into a lower-dimensional representation.

### Combining Features
- **Concatenating Features**: Merged scaled numerical features and encoded categorical features to form comprehensive feature sets.

### Enhanced XGBoost Model
- **Training**: Trained an XGBoost classifier on the combined feature set.
- **Evaluation**: Assessed model accuracy on test and validation sets.
- **Feature Importance Analysis**: Identified and visualized the top 5 features contributing to anomaly detection.

### Anomaly Visualization
- **Feature Importance Plot**: Created bar plots to display the importance of the top features.
- **Anomaly Examples**: Displayed examples of data from the top features where anomalies were detected.

## Summary

Through these contributions, I have:
- Implemented robust preprocessing and data transformation techniques.
- Applied and evaluated multiple anomaly detection methods.
- Used dimensionality reduction for effective visualization.
- Integrated autoencoder techniques to enhance feature extraction.
- Trained and validated an XGBoost model for improved anomaly detection.
- Provided comprehensive visual tools to analyze anomalies and feature importances.

These efforts have significantly enhanced our project's ability to detect, understand, and visualize anomalies in the dataset.
