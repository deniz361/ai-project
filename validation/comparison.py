import itertools
import matplotlib.pyplot as plt
import numpy as np
from results import Results

def compare_models(predictions_files, model_names):
    assert len(predictions_files) == len(model_names), "Number of files should match number of model names"
    
    metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }
    
    for file in predictions_files:
        results = Results(file)
        
        accuracy = results.accuracy()
        precision = results.precision()
        recall = results.recall()
        f1 = results.f1_score()
        
        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1 Score'].append(f1)
    
    # Print comparison table
    print("Comparison of Models:")
    for i, model_name in enumerate(model_names):
        print(f"{model_name}:")
        print(f"Accuracy: {metrics['Accuracy'][i]:.2f}")
        print(f"Precision: {metrics['Precision'][i]:.2f}")
        print(f"Recall: {metrics['Recall'][i]:.2f}")
        print(f"F1 Score: {metrics['F1 Score'][i]:.2f}")
        print()
        
        # Plot horizontal bar chart for metrics
        fig, ax = plt.subplots(figsize=(8, 3))
        metrics_values = [metrics['Accuracy'][i], metrics['Precision'][i], metrics['Recall'][i], metrics['F1 Score'][i]]
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        ax.barh(metrics_labels, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_xlim(0, 1)  # Set limit for x-axis to 0-1 for scores
        ax.set_title(f'Metrics for {model_name}')
        ax.set_xlabel('Score')
        ax.invert_yaxis()  # Invert y-axis to have Accuracy at the top
        
        # Add scores as text on bars
        for i, v in enumerate(metrics_values):
            ax.text(v + 0.02, i, f'{v:.2f}', va='center', color='black', fontweight='bold')
        
        plt.show()
    
    # Plot confusion matrix for each model
    for i, file in enumerate(predictions_files):
        results = Results(file)
        cm = results.confusion_matrix()
        
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {model_names[i]}")
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(results.data['anomaly'])))
        plt.xticks(tick_marks, ['0', '1'], rotation=45)
        plt.yticks(tick_marks, ['0', '1'])
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

if __name__ == "__main__":
    predictions_files = ['algorithms/NN/results/test_predictions.csv', 'algorithms/RFC/results/test_predictions.csv']
    model_names = ['Neural Network', 'Random Forest']
    
    compare_models(predictions_files, model_names)
