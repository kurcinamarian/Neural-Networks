import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#number of epochs
num_epochs = 30

#model definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            #making 28*28 into vector
            nn.Flatten(),
            #first layer (input to 100)
            nn.Linear(28 * 28, 100),
            #relu activation function
            nn.ReLU(),
            #second layer (100 to 50)
            nn.Linear(100, 50),
            #relu activation function
            nn.ReLU(),
            #third layer (50 to output)
            nn.Linear(50, 10),
        )

    #gets predicted output from running input through layers
    def forward(self, x):
        return self.layers(x)

#training the model
def train_model(optimizer_type, learning_rate, momentum):
    #create model
    model = MLP()
    #from output creates probabilities of items
    criterion = nn.CrossEntropyLoss()

    #optimizer type
    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), learning_rate)
    elif optimizer_type == "SGD_momentum":
        optimizer = optim.SGD(model.parameters(), learning_rate, momentum)
    elif optimizer_type == "ADAM":
        optimizer = optim.Adam(model.parameters(), learning_rate)
    else:
        raise ValueError("Unknown optimizer type")
    # Lists to store losses and accuracies
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        #sets model to training
        model.train()
        running_train_loss = 0.0
        #trains for all images in training dataset
        for images, labels in train_loader:
            #resets gradients
            optimizer.zero_grad()
            #model's prediction
            outputs = model(images)
            #difference between prediction and true outputs
            loss = criterion(outputs, labels)
            #gradients computing
            loss.backward()
            #updating weights based on optimizer
            optimizer.step()
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        #test model
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        #compute variables needed for graph
        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        print(f"{optimizer_type} - Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")
    #create graph
    fig, ax1 = plt.subplots(figsize=(10, 6))
    #training loss
    ax1.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', marker='o')
    #testing loss
    ax1.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='red', marker='o')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Test Loss - {optimizer_type}')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    #second axis
    ax2 = ax1.twinx()
    #accuracies
    ax2.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='orange', marker='x')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='lower right')
    plt.show()
    #return trained model
    return model

#testing the model
def test_model(model):
    #sets model to evaluation mode
    model.eval()
    correct = 0
    total = 0
    #confusion matrix ()
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    #disables gradient computing
    with torch.no_grad():
        #tests for all images in test dataset
        for images, labels in test_loader:
            #model's prediction
            outputs = model(images)
            #finds max in output that being the predicted number
            _, predicted = torch.max(outputs.data, 1)
            #adds number of data in batch
            total += labels.size(0)
            #for images in batches add number of correctly labeled
            correct += (predicted == labels).sum().item()
            #update of the confusion matrix (rows are true labels columns are predicted labels)
            for p, t in zip(predicted, labels):
                matrix[t.item()][p.item()] += 1
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    #print confusion matrix
    print("Confusion Matrix:")
    print("Predicted->  " + "  ".join(f"{i:>5}" for i in range(10)))
    print("------------------------------------------------------------------------------------------------------------------")
    #print rows with true values on left
    for i, row in enumerate(matrix):
        print(f"\t\t{i:<3} |", "  ".join(f"{x:5}" for x in row))
    return accuracy

#testing the parameters
l_r = float(input("Learning rate:"))
momentum = float(input("Momentum for SGD_momentum:"))

#transformation of data
transform = transforms.Compose([
    transforms.ToTensor(),
])

#loads train data and transform them
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
#loads test data and transform them
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

#splits data into batches and shuffles them
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#splits data into batches
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#tries to train and then test model for optimization types

print(f"\nTraining with SGD")
#train model with SGD
model = train_model("SGD",0.1,0.9)
#test model
test_accuracy_SGD = test_model(model)
print(f"\nTraining with SGD_momentum")
# train model with SGD_momentum
model = train_model("SGD_momentum", 0.1, 0.9)
# test model
test_accuracy_SGD_momentum = test_model(model)
print(f"\nTraining with ADAM")
# train model with ADAM
model = train_model("ADAM", 0.01, 0.9)
# test model
test_accuracy_ADAM = test_model(model)