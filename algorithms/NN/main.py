# main.py

import torch
import torch.optim as optim
from dataloader import load_data
from algorithm import SimpleNN

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).view(-1, 1)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = (output >= 0.5).float()
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

def main():
    # Parameters
    filepath = '/Users/awthura/THD/ai-project/datasets/supervised_dataset.csv'
    input_size = 30  # Change this based on your dataset features
    learning_rate = 0.001
    epochs = 20
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader = load_data(filepath)
    
    # Initialize model, loss function, and optimizer
    model = SimpleNN(input_size=input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{epochs} completed')
    
    # Test the model
    test(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()
