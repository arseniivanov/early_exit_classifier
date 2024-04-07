import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

def load_model(model_class, model_path, device):
    model = model_class().to(device)  # Instantiate the model and move it to the device
    model.load_state_dict(torch.load(model_path))  # Load the saved state dictionary
    model.eval()  # Set the model to evaluation mode
    return model


# Define the early exit network
class EarlyExitNet(nn.Module):
    def __init__(self):
        super(EarlyExitNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x8x8
        )

        self.cont = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 128x4x4
        )

        #self.cont2 = nn.Sequential(
        #    nn.Linear(128*4*4, 64)
        #)

        self.exit0 = nn.Linear(64*8*8, 10)
        self.exit1 = nn.Linear(128*4*4, 10)
        #self.exit2 = nn.Linear(64,10)

    def forward(self, x):
        x = self.features(x)
        first_exit = x.view(x.size(0), -1)

        exit1_out = self.exit0(first_exit)

        x = self.cont(x)
        second_exit = x.view(x.size(0), -1)
        exit2_out = self.exit1(second_exit)

        return exit1_out, exit2_out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Initialize the network
net = EarlyExitNet().to(device)

#Optimizer
optimizer = optim.AdamW(net.parameters(), lr=0.001)

# Data transformations and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

import torch.nn.functional as F
def combined_loss(outputs, labels, weights, beta):
    criterion = nn.CrossEntropyLoss()
    confidences = [torch.max(F.softmax(output, dim=1), dim=1)[0] for output in outputs]
    confidence_penalties = [beta * (1 - confidence.mean()) for confidence in confidences]

    total_loss = 0
    for output, weight, penalty in zip(outputs, weights, confidence_penalties):
        loss = criterion(output, labels)
        weighted_loss = weight * loss
        penalized_loss = weighted_loss + penalty
        total_loss += penalized_loss
    
    return total_loss


def train(model, train_loader, optimizer, epochs=5, weights=[2, 1], beta=0.1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()

            outputs = model(images)
            loss = combined_loss(outputs, labels, weights, beta)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}')

    print('Finished Training')

# Call the training function
if Path("early_exit.state").exists():
    net = load_model(EarlyExitNet, 'early_exit.state', device)
else:
    train(net, trainloader, optimizer, epochs=20)
    torch.save(net.state_dict(), "early_exit.state")  # Save the model state to a file

# Function to evaluate the model on the test set
def evaluate_model(model, test_loader, device, confidence_threshold=0.9):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    exit_counts = [0, 0]  # Counts for exits 1 and 2

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            confidences = [F.softmax(output, dim=1).max(1) for output in outputs]  # Confidence and predictions
            
            # Decide on exits based on confidence and track their use
            for i, (confidence, prediction) in enumerate(confidences):
                # Check if confidence exceeds the threshold and hasn't been counted yet
                if i == 0:  # First exit specifics
                    exit_taken = confidence >= confidence_threshold
                else:  # Ensuring that samples not taken by the first exit are evaluated in the second
                   #exit_taken = (confidence >= confidence_threshold) & (confidences[i-1][0] < confidence_threshold)
                   exit_taken = ~exit_taken

                exit_counts[i] += exit_taken.sum()
                correct += (prediction[exit_taken] == labels[exit_taken]).sum().item()
            
            total += labels.size(0)
    
    print("Amount of labels:", total)
    accuracy = 100 * correct / total
    return accuracy, exit_counts


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


# Evaluate the model
accuracy, exit_counts = evaluate_model(net, testloader, device)
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Exit counts: {exit_counts}")