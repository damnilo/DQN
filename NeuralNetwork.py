import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cpu")   

# Definisanje transformacije za normalizaciju podataka
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Preuzimanje i učitavanje trening i test skupa
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Kreiranje dataloadera za trening i test skup
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for images, labels in test_loader:
    print(f"Shape of images : {images.shape}") # shape: [B, C, H, W]
    print(f"Shape of labels: {labels.shape} {labels.dtype}")
    break

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, device, train_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Compute prediction and loss
        pred = model(images)
        loss = loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

for epoch in range(10):
    avg_loss = train(model, device, train_loader, loss_fn, optimizer)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    batch_accuracies = []
    batch_losses = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            total += loss.item()
            batch_losses.append(loss.item())

            correct = (outputs.argmax(dim=1) == labels).float().sum().item()
            total += labels.size(0)
            batch_accuracies.append(correct / labels.size(0))

    accuracy = sum(batch_accuracies) / len(batch_accuracies)
    average_loss = total / len(test_loader)

    print(f"Test Accuracy: {accuracy:.4f}, Average Loss: {average_loss:.4f}")
    return batch_accuracies, batch_losses

eval_res = evaluate(model, test_loader)