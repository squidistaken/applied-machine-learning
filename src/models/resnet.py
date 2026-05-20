import torch
import torch.nn as nn
import torch.optim as optim

from base import BaseModel
from torchvision.models import resnet18
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))     
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
        


class resnet(BaseModel, nn.Module):
    def __init__(self, dataset: ChestXRayDatasetPyTorch, learning_rate: float = 0.001):
        BaseModel.__init__(self, dataset)
        nn.Module.__init__(self)

        self.num_classes = len(self.classes)
        self.in_channels = 1

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 16, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def forward_pass(self, x):
        """
        predict class of input image.
        """

        self.eval()

        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs, dim=1)

        return predictions

    def backward_pass(self, x_train, y_train, epochs: int = 10, **kwargs):
        """
        train the CNN model

        the arguments are:
            x_train: training images
            y_train: training labels
            epochs: number of times the model sees the training data
        """

        self.train()

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            outputs = self.forward(x_train)
            loss = self.loss_function(outputs, y_train)

            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    def evaluate(self, x_test, y_test):
        """
        Test the performance of the model.

        Returns: FIXME should be macro f1 score, precision, recall and confusion matrix
            accuracy
        """

        self.eval()

        with torch.no_grad():
            outputs = self.forward(x_test)
            predictions = torch.argmax(outputs, dim=1)

            correct = (predictions == y_test).sum().item()
            total = y_test.size(0)

            accuracy = correct / total

        return accuracy
    
    def _save_weights(self, path):
        """
        Save model weights.
        """

        torch.save(self.state_dict(), path)

    def _load_weights(self, path):
        """
        Load a trained model.
        """

        self.load_state_dict(torch.load(path))
        self.eval()
    

    






