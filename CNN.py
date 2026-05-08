import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

device = torch.device("cpu")

TRAIN_BATCH_SIZE = 128
NUM_EPOCHS = 5
lr = 1e-3
WEIGHT_DECAY = 1e-4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x))

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
model = CNN().to(device)
print(model)

def loss_fn(logits, labels):
    one_hot_labels = F.one_hot(labels, num_classes=10).float()
    cross_entropy = -torch.sum(one_hot_labels * F.log_softmax(logits, dim=1)) / labels.shape[0]
    l2_loss = 0.5 * sum(torch.sum(param ** 2) for param in model.parameters())
    return cross_entropy + WEIGHT_DECAY * l2_loss

def train(model, loader, optimizer):
    model.train()
    epoch_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()

    return correct / len(loader.dataset)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

test_accuracies = []
epoch_losses = []

for epoch in range(NUM_EPOCHS):
    epoch_loss = train(model, train_loader, optimizer)
    test_accuracy = evaluate(model, test_loader)

    epoch_losses.append(epoch_loss)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")