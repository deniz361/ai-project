import torch
import torch.optim as optim
from dataloader import load_data
from algorithm import SimpleNN
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import shap

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()

def test(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device).view(-1, 1)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = (output >= 0.5).float()
            correct += pred.eq(target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    print(f'Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # Calculate precision, recall, and f1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(cm)

def feature_importance(model, val_loader, device, feature_names):
    model.eval()
    data_iter = iter(val_loader)
    data, target = next(data_iter)  # Use next(data_iter) instead of data_iter.next()
    data, target = data.to(device), target.to(device).view(-1, 1)
    
    # Explain the model's predictions using SHAP
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data, check_additivity=False)
    
    # Plot SHAP values
    shap.summary_plot(shap_values, data.cpu().numpy(), feature_names=feature_names, plot_type='bar')

def main():
    # Parameters
    train_filepath = 'datasets/split/supervised_learning_dataset/train.csv'
    val_filepath = 'datasets/split/supervised_learning_dataset/val.csv'
    input_size = 23  # Change this based on your dataset features
    learning_rate = 0.001
    epochs = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Load data and get feature names
    train_loader, val_loader, feature_names = load_data(train_filepath, val_filepath)
    
    # Initialize model, loss function, and optimizer
    model = SimpleNN(input_size=input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer, device, epoch)
        print(f'Epoch {epoch+1}/{epochs} completed')
    
    # Validate the model
    test(model, val_loader, criterion, device)
    
    # Save model weights
    torch.save(model.state_dict(), 'algorithms/NN/weights/model_weights.pth')
    
    ## This line is for feature importance. Could be commented out for normal training
    #Explain the model feature importance using SHAP
    #feature_importance(model, val_loader, device, feature_names=feature_names)

if __name__ == '__main__':
    main()

