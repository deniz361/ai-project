import torch
import pandas as pd
from algorithm import SimpleNN
from inference_dataloader import load_inference_data

def load_model(model_path, input_size):
    model = SimpleNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_inference(model, data_loader, device):
    model.to(device)
    all_preds = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
    return all_preds

def main(filepath: str, model_path: str):
    input_size = 23  # Ensure this matches your dataset features after preprocessing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_loader = load_inference_data(filepath)
    
    # Load the original data to align predictions with it
    original_data = pd.read_csv(filepath)
    original_data = original_data.drop(columns=['Material number', 'Plant'])
    
    # Load model
    model = load_model(model_path, input_size)

    # Run inference
    predictions = run_inference(model, data_loader, device)

    # Save predictions to a CSV file
    original_data['Prediction'] = predictions
    original_data.to_csv('algorithms/NN/results/predictions.csv', index=False)

    # Save anomalies to a CSV file
    anomalies = original_data[original_data['Prediction'] == 1]
    anomalies.to_csv('algorithms/NN/results/anomalies.csv', index=False)

    print("Predictions saved to predictions.csv")
    print("Anomalies saved to anomalies.csv")

if __name__ == '__main__':
    main("datasets/split/supervised_learning_dataset/test.csv", "algorithms/NN/weights/model_weights.pth")
