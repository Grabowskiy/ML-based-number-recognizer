import numpy as np
import torch
import torch.cuda
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#Configuring the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}!")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#Loading data
train_data = datasets.MNIST('./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST('./data', train=False, transform=transform, download=True)

#Batching
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        #Initialising the model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128, True),
            nn.ReLU(),
            nn.Linear(128, 64, True),
            nn.ReLU(),
            nn.Linear(64, 10, True),
        )

    def forward(self, x):
        return self.model(x)

def load_model():
    model = NN().to(device)
    state_dict = torch.load("mnist_torch_model.pth", map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_model(lr, epochs):
    #Making the instance of NN class
    model = NN().to(device)

    #Initialising the optimizer and loss functions
    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    min_loss = None
    #Starting training
    model.train()
    for epoch in range(epochs):
        current_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            current_loss += loss.item()
        average_loss = current_loss / len(train_loader)
        print(f'{epoch+1}/{epochs} epoch, Train Loss: {average_loss:.4f}')

        #Evaluating the model
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                if min_loss == None or min_loss[0] > loss:
                    min_loss = (loss, epoch+1)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += loss.item()
        accuracy = correct / total * 100
        test_loss = test_loss / len(test_loader)
        print(f'The test accuracy of the model over {epoch+1} epochs: {accuracy}% and test loss: {test_loss}')

    print(f"Min test loss: {min_loss[0]} was at epoch {min_loss[1]}")
    torch.save(model.state_dict(), "mnist_torch_model.pth")

    return model

if __name__ == "__main__":
    train_model(0.001, 10)
