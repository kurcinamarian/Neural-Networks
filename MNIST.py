import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 128
learning_rate = 0.05
num_epochs = 30
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
        )

    def forward(self, x):
        return self.layers(x)


def train_model(optimizer_type):
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.05)
    elif optimizer_type == "SGD_momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    elif optimizer_type == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError("Unknown optimizer type")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"{optimizer_type} - Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    return model


def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for p, t in zip(predicted, labels):
                matrix[t.item()][p.item()] += 1

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    for row in matrix:
        print(row)
    return accuracy



for optimizer_type in ["SGD", "SGD_momentum", "ADAM"]:
    print(f"\nTraining with {optimizer_type}")
    trained_model = train_model(optimizer_type)
    test_accuracy = evaluate_model(trained_model)
