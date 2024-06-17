import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class Results:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Check and convert 'Prediction' column if necessary
        self._convert_prediction_column()
    
    def _convert_prediction_column(self):
        if isinstance(self.data['Prediction'].iloc[0], str):  # Check if first element is a string
            try:
                self.data['Prediction'] = self.data['Prediction'].apply(lambda x: float(x.strip('[]')))
            except ValueError:
                pass  # Handle cases where conversion fails (already in numeric format)
    
    def confusion_matrix(self):
        predictions = self.data['Prediction']
        true_labels = self.data['anomaly']
        cm = confusion_matrix(true_labels, predictions.round().astype(int))  # Ensure predictions are integers
        return cm
    
    def accuracy(self):
        predictions = self.data['Prediction']
        true_labels = self.data['anomaly']
        acc = accuracy_score(true_labels, predictions.round().astype(int))
        return acc
    
    def precision(self):
        predictions = self.data['Prediction']
        true_labels = self.data['anomaly']
        prec = precision_score(true_labels, predictions.round().astype(int))
        return prec
    
    def recall(self):
        predictions = self.data['Prediction']
        true_labels = self.data['anomaly']
        rec = recall_score(true_labels, predictions.round().astype(int))
        return rec
    
    def f1_score(self):
        predictions = self.data['Prediction']
        true_labels = self.data['anomaly']
        f1 = f1_score(true_labels, predictions.round().astype(int))
        return f1
    
    def plot_pie_chart(self):
        predictions = self.data['Prediction']
        true_labels = self.data['anomaly']
        
        correct_predictions = np.sum(predictions.round().astype(int) == true_labels)
        incorrect_predictions = np.sum(predictions.round().astype(int) != true_labels)
        
        labels = ['Correct Predictions', 'Incorrect Predictions']
        sizes = [correct_predictions, incorrect_predictions]
        colors = ['#1f77b4', '#ff7f0e']
        
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        #plt.title('Pie Chart for Predictions')
        plt.axis('equal')
        plt.show()

