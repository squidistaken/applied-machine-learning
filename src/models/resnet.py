import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.models.base import BaseModel
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from src.models.cnn import CNN


class ResNet(CNN):
    """
    Pretrained ResNet-18 Class for X-Ray Image Classification.
    """

    def __init__(
        self,
        dataset: ChestXRayDatasetPyTorch,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> None:
        """
        Initialise the class.

        Args:
            dataset (ChestXRayDatasetPyTorch): The dataset to train on.
            learning_rate (float): The learning rate for training. Defaults to
                                   0.001.
            weight_decay (float): The weight decay (L2 penalty). Defaults to
                                  1e-4.
        """
        BaseModel.__init__(self, dataset)
        nn.Module.__init__(self)

        self.num_classes = len(self.classes)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # To prevent overfitting, we freeze the pretrained core layers.
        for param in self.resnet.parameters():
            param.requires_grad = False

        # As our dataset is in grayscale, our first layer needs to accept 1
        # channel. Replacing the layer automatically sets requires_grad=True
        # for this layer.
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Average the weights of the 3 channels to adapt to the grayscale channel
        with torch.no_grad():
            self.resnet.conv1.weight.copy_(
                original_conv1.weight.mean(dim=1, keepdim=True)
            )

        # To enable Monte Carlo (MC) Dropout on ResNet, we must append a
        # Dropout layer right before the final classification linear layer.
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(num_ftrs, self.num_classes)
        )

        self.loss_function = nn.CrossEntropyLoss()
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: An output tensor.
        """
        return self.resnet(x)
