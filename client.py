from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import flwr as fl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- MODEL --
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# -------------------- DATA LOADER --------------------
def load_data(path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader

# -------------------- TRAIN --------------------
def train(model, trainloader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# -------------------- TEST --------------------
def test(model, testloader):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_total = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return loss_total / len(testloader), 100 * correct / total

# -------------------- FLOWER CLIENT --------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": acc}

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    train_path = os.path.join(args.dataset, "train")
    test_path  = os.path.join(args.dataset, "test")

    print("Using training data from:", train_path)
    print("Using testing data from:", test_path)

    trainloader = load_data(train_path)
    testloader = load_data(test_path)

    model = PneumoniaModel().to(DEVICE)
    client = FlowerClient(model, trainloader, testloader)

    fl.client.start_client(server_address="127.0.0.1:8081", client=client.to_client())
    
if __name__ == "__main__":
    main()

def fit(self, parameters, config):
    print(">>> CLIENT RECEIVED FEDERATED ROUND:", config["server_round"])
    self.set_parameters(parameters)
    train(self.model, self.trainloader)
    return self.get_parameters(config), len(self.trainloader.dataset), {}
