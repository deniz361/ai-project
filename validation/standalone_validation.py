from results import Results

def standalone_validation(predictions_file):
    results = Results(predictions_file)
    
    cm = results.confusion_matrix()
    acc = results.accuracy()
    prec = results.precision()
    rec = results.recall()
    f1 = results.f1_score()
    results.plot_pie_chart()

    print(f"Standalone Validation Results for {predictions_file}:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print()

if __name__ == "__main__":
    predictions_files = ['algorithms/NN/results/test_predictions.csv', 'algorithms/RFC/results/test_predictions.csv']
    
    for file in predictions_files:
        standalone_validation(file)
