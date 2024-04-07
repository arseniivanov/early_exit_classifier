import torchvision
import torchvision.transforms as transforms

def download_cifar10():
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing the dataset
    ])

    # Download the training data
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Download the test data
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    print("CIFAR-10 training and test datasets downloaded and cached.")

if __name__ == "__main__":
    download_cifar10()
