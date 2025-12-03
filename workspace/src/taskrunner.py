# src/taskrunner.py
from openfl.component.task_runner import TaskRunner
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Net


class TemplateTaskRunner(TaskRunner):
    """Custom Task Runner for Federated Learning."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Net()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_task(self, col_data, **kwargs) -> Dict[str, Any]:
        """Perform one round of local training."""
        self.model.train()
        total_loss = 0
        correct = 0
        for data, target in col_data:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

        avg_loss = total_loss / len(col_data)
        acc = correct / (len(col_data.dataset))
        return {"loss": avg_loss, "acc": acc}

    def validate_task(self, col_data, **kwargs) -> Dict[str, Any]:
        """Perform validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in col_data:
                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        avg_loss = total_loss / len(col_data)
        acc = correct / len(col_data.dataset)
        return {"loss": avg_loss, "acc": acc}

