import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base import BaseModel
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch


class CNN(BaseModel, nn.Module):
    """
    simple CNN model for chest X-ray image classification.

    the architecture is supposed to be:
    - layer 1: convolutional layer
    - layer 2: convolutional layer
    - layer 3: fully connected layer

    """

    def __init__(
        self, dataset: ChestXRayDatasetPyTorch, learning_rate: float = 0.001
    ):
        BaseModel.__init__(self, dataset)
        nn.Module.__init__(self)

        self.num_classes = len(self.classes)

        # the layer 1 is the first convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )

        # the layer 2 is the second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        # extra operations used between layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # this makes the model work even if the image size changes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # the layer 3 is a fully connected layer
        self.fc = nn.Linear(32, self.num_classes)

        # training setup
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        PyTorch forward method

        input shape is as follows:
        x = (batch_size, 1, height, width)

        output:
        class scores for each image
        """

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.global_pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return x

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
