import torch
import torch.nn as nn

from base import BaseModel
from torchvision.models import resnet18
from src.data.dataset import ChestXRayDataset

class resnet(BaseModel, nn.Module):
    def __init__(self, dataset: ChestXRayDataset, learning_rate: float = 0.001):
        BaseModel.__init__(self, dataset)
        nn.Module.__init__(self)

        self.model = resnet18(weights='DEFAULT')