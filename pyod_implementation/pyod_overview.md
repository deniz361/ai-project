# Open questions from last meeting:
- Gesamtwert:
  - A lot of values are equal to zero: Are these just items that are not in stock?
  - A few Gesamtwert values are very large: Were these special deliveries/products?
- Preiseinheit:
  - What does Preiseinheit 0 mean? Is it some kind of service?
  - What does it do in general? Just for easier formatting?



# PyOD: Python Outlier Detection

## Overview
PyOD, short for Python Outlier Detection, is a comprehensive Python library for outlier detection. It provides a wide range of algorithms and tools to identify outliers in multivariate data. PyOD is designed with simplicity, flexibility, and ease of use in mind, making it suitable for both beginners and advanced users in the field of anomaly detection.

## Key Features
- **Diverse Collection of Algorithms**: PyOD offers a diverse collection of outlier detection algorithms, covering various techniques and methodologies. Some of the notable algorithms included in PyOD are:
  - **Angle-Based Outlier Detection (ABOD)**: Measures the angle between each point and its neighbors to detect outliers.
  - **k-Nearest Neighbors (kNN)**: Identifies outliers based on the distance to their k-nearest neighbors.
  - **Isolation Forest (IF)**: Constructs isolation trees to isolate outliers more efficiently than traditional methods.
  - **Minimum Covariance Determinant (MCD)**: Estimates the minimum covariance determinant to detect outliers.
  - **One-Class SVM (OCSVM)**: Constructs a hyperplane to separate normal data points from outliers in a high-dimensional space.
  - **Local Outlier Factor (LOF)**: Computes the local density deviation of a data point with respect to its neighbors to detect outliers.
  - **Histogram-Based Outlier Score (HBOS)**: Utilizes histograms to compute the outlier score of data points.

- **Unified Interface**: All algorithms in PyOD follow a consistent interface, making it easy to switch between different methods without significant changes to the code.

- **Scalability**: PyOD is designed to handle large datasets efficiently, allowing users to perform outlier detection on massive datasets with relative ease.

- **Visualization Tools**: PyOD provides built-in visualization tools to visualize outlier scores, decision boundaries, and detected outliers, aiding in the interpretation and analysis of results.

- **Model Evaluation**: The library includes utilities for evaluating the performance of outlier detection models, such as ROC (Receiver Operating Characteristic) curve analysis, precision-recall curve analysis, and various metrics for quantifying model performance.


## List of models
