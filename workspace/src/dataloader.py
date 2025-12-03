import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class PyTorchMNISTInMemory:
    def __init__(self, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Download MNIST dataset
        dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        self.trainset, self.valset = random_split(dataset, [55000, 5000])

    def get_train_loader(self, batch_size=64):
        return DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

    def get_valid_loader(self, batch_size=64):
        return DataLoader(self.valset, batch_size=batch_size, shuffle=False)

