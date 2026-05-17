from models.base import BaseModel
from data.dataset_pytorch import ChestXRayDatasetPyTorch
from data.dataset_lightgbm import ChestXRayDatasetLightGBM
from typing import Optional
from torch.utils.data import DataLoader
import matplotlib.pyplot  as plt
import numpy as np

# Define plotting style.
plt.style.use("seaborn-v0_8-dark-palette")
plt.rcParams.update(
    {
        "figure.figsize": (12, 6),
        "axes.labelsize": 16,
        "axes.grid": True,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 18,
        "legend.fontsize": 16,
        "lines.linewidth": 2,
        "text.usetex": False,
        "font.family": "serif",
        "image.cmap": "magma",
    }
)

class Trainer:
    """Class for a trainer for image classification models"""
    
    def __init__(self, model: BaseModel, 
                 train_data: ChestXRayDatasetLightGBM | ChestXRayDatasetPyTorch, 
                 eval_data: ChestXRayDatasetLightGBM | ChestXRayDatasetPyTorch,
                 test_data: Optional[ChestXRayDatasetPyTorch | ChestXRayDatasetLightGBM] = None,
                 batch_size: int = 32):
        """Initialize the trainer
        
        Args:
            model (Basemodel): the model
            train_data (ChestXRayDatasetLightGBM | ChestXRayDatasetPyTorch): training data
            eval_data (ChestXRayDatasetLightGBM | ChestXRayDatasetPyTorch): evaluation data
            test_data (Optional[ChestXRayDatasetLightGBM | ChestXRayDatasetPyTorch]): training data, defaults to None
            batch_size (int): batch size, defaults to 32
        """
        self.model = model
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            train_data, 
            batch_size=batch_size,
            shuffle=True
        )
        
        self.eval_loader = DataLoader(
            eval_data,
            batch_size=batch_size,
            shuffle=False
        )
        
        if test_data is not None:
            self.test_loader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False
            )
        
    def reset_history(self):
        """Reset the training history"""
        self.history = {"train_loss": [], "eval_loss": []}
    
    def train()
        ...
    
    def evaluate():
        ...
        
    def plot_history(
        self, show: bool = True, save_path: Optional[str] = None
    ) -> None:
        """Plot the training and evaluation loss history.

        Args:
            show (bool, optional): Whether to show the plot immediately.
                                   Defaults to True.
            save_path (Optional[str], optional): The path to save the plot to.
                                                 Defaults to None.
        """
        # Create x-axis values for train and eval loss based on the number of recorded losses.
        x_train = np.asarray(range(1, len(self.history["train_loss"]) + 1))
        x_eval = np.asarray(range(1, len(self.history["eval_loss"]) + 1))
        x_train = x_train * (
            len(self.history["eval_loss"]) / len(self.history["train_loss"])
        )

        plt.figure()
        plt.plot(x_train, self.history["train_loss"], label="Train Loss")
        plt.plot(x_eval, self.history["eval_loss"], label="Eval Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss History")
        plt.legend()
    
        if save_path:
                plt.savefig(save_path)
                LOGGER.info(f"Training history plot saved to {save_path}")

        if show:
            plt.show()
    
    def save_model():
        ...
        
    def load_model():
        ...