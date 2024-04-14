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

        f_1 = 128
        f_2 = 256
        f_3 = 512
        f_4 = 64

        self.features = nn.Sequential(
            nn.Conv2d(3, f_1, kernel_size=3, padding=1), #3x32x32
            nn.BatchNorm2d(f_1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # 32x16x16
            nn.Conv2d(f_1, f_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f_2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # 64x8x8
        )

        self.cont = nn.Sequential(
            nn.Conv2d(f_2, f_3, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)   # 128x4x4
        )

        self.cont2 = nn.Sequential(
           nn.Linear(f_3*4*4, f_4)
        )

        self.exit0 = nn.Linear(f_2*8*8, 10)
        self.exit1 = nn.Linear(f_3*4*4, 10)
        self.exit2 = nn.Linear(f_4,10)


        self.res = nn.Sequential(        
            nn.MaxPool2d(4, 4),  # 3x32x32 -> 3x8x8
            nn.Conv2d(3, f_2, kernel_size=1),
            nn.LeakyReLU()
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(3, f_2, kernel_size=9, stride=3),
            nn.LeakyReLU()
        )
        self.leaky = nn.LeakyReLU()

        self.num_outputs = 3

    def forward(self, x):

        id = x

        x = self.features(x)
        x += self.res(id)
        x += self.spatial(id)
        x = self.leaky(x)

        first_exit = x.view(x.size(0), -1)

        exit1_out = self.exit0(first_exit)

        x = self.cont(x)
        second_exit = x.view(x.size(0), -1)
        exit2_out = self.exit1(second_exit)

        x = self.cont2(second_exit)

        third_exit = x.view(x.size(0), -1)
        exit3_out = self.exit2(third_exit)

        return exit1_out, exit2_out, exit3_out

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


def train(model, train_loader, optimizer, epochs=5, weights=[2.5, 2, 1], beta=0.1):
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
    train(net, trainloader, optimizer, epochs=25)
    torch.save(net.state_dict(), "early_exit.state")  # Save the model state to a file


def evaluate_model(model, test_loader, device, confidence_threshold=0.9):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    exit_counts = [0] * model.num_outputs

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # List of tuples of confidence values and predictions for each exit
            confidences = [F.softmax(output, dim=1).max(1) for output in outputs]
            
            # Tracker for whether a sample has been taken by any exit yet
            taken_by_any_exit = torch.zeros(labels.size(0), device=device, dtype=torch.bool)

            # Iterate over all outputs (exits) except the last one
            for i, (confidence, prediction) in enumerate(confidences[:-1]):
                exit_taken = (confidence >= confidence_threshold) & (~taken_by_any_exit)
                taken_by_any_exit |= exit_taken

                # Update the counts and correctness for this exit
                exit_counts[i] += exit_taken.sum().item()
                correct += (prediction[exit_taken] == labels[exit_taken]).sum().item()

            # Ensure the last exit takes any remaining samples
            last_exit = ~taken_by_any_exit
            final_confidence, final_prediction = confidences[-1]
            exit_counts[-1] += last_exit.sum().item()
            correct += (final_prediction[last_exit] == labels[last_exit]).sum().item()

            total += labels.size(0)

    print("Total labels processed:", total)
    accuracy = 100 * correct / total
    return accuracy, exit_counts



testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


# Evaluate the model
accuracy, exit_counts = evaluate_model(net, testloader, device)
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Exit counts: {exit_counts}")


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def individual_exit_grid_search(model, loader, device, base_threshold=0.9, threshold_range=(0.5, 0.99), steps=25):
    model.eval()
    num_exits = model.num_outputs  # Assuming model.outputs provides the number of exits
    optimal_thresholds = [base_threshold] * num_exits
    base_accuracy = evaluate_model_with_thresholds(model, loader, device, optimal_thresholds)

    for exit_index in range(num_exits):
        best_exit_accuracy = 0
        best_threshold = base_threshold
        thresholds = torch.linspace(*threshold_range, steps)

        for threshold in thresholds:
            current_thresholds = optimal_thresholds[:]
            current_thresholds[exit_index] = threshold.item()
            accuracy = evaluate_model_with_thresholds(model, loader, device, current_thresholds)
            
            if accuracy > best_exit_accuracy:
                best_exit_accuracy = accuracy
                best_threshold = threshold.item()

        optimal_thresholds[exit_index] = best_threshold
        print(f"Optimized Threshold for Exit {exit_index}: {best_threshold}, Achieved Accuracy: {best_exit_accuracy}%")

    overall_accuracy = evaluate_model_with_thresholds(model, loader, device, optimal_thresholds)
    print(f"Overall Model Accuracy with Optimized Thresholds: {overall_accuracy}%")
    return optimal_thresholds

def evaluate_model_with_thresholds(model, loader, device, thresholds):
    total, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            taken = torch.zeros(labels.size(0), dtype=torch.bool, device=device)

            for output, threshold in zip(outputs, thresholds):
                confidence = torch.max(F.softmax(output, dim=1), dim=1)[0]
                take = (confidence >= threshold) & (~taken)
                taken |= take
                correct += (output.max(1)[1][take] == labels[take]).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

# Example Usage
# Assuming 'net', 'device', and 'testloader' are defined
# optimized_thresholds = individual_exit_grid_search(net, testloader, device)