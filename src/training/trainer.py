from models.base import BaseModel
from data.dataset_pytorch import ChestXRayDatasetPyTorch
from data.dataset_lightgbm import ChestXRayDatasetLightGBM
from typing import Optional
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot  as plt
import numpy as np
from src.constants import LOGGER
from pathlib import Path
from rich.progress import track

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
    
    def train(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        patience: int = 3,
    ) -> None:
        """Train the model.

        Args:
            num_epochs (int, optional): The number of epochs to train for.
                                        Defaults to 5.
            learning_rate (float, optional): The learning rate to use for
                                             optimization. Defaults to 1e-3.
            patience (int, optional): The number of epochs to wait for
                                      improvement, if early stopping is
                                      enabled. Defaults to 3.
        """
        self.reset_history()
        
        #TODO set up optimizer and loss functions? Not generalized in basemodel class yet
            
        for epoch in track(range(num_epochs), description="Training epochs"):
            total_loss = 0
            for batch in track(
                self.train_loader, description="Training batches"
            ):
                # update model parameters based on the batch
                #TODO get the train_x inputs and train_y labels corresponding to the batch
                
                #TODO set gradients to zero
                
                #TODO calculate outputs and loss
                
                #TODO do a backwards pass using the loss and outputs (?)
                
                #TODO append training loss to the self.history dictionary
                #TODO add training loss to total_loss
                
                avg_loss = total_loss / len(self.train_loader)
                #TODO calculate the eval_loss using self.evaluate()
                eval_loss =self.evaluate(...)
                
                #TODO append evaluation loss to the self.history dictionary
                
                LOGGER.infont(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}"
                )
                ...
                
    
    def evaluate(self, use_test: bool = False) -> float:
        """Evaluate the model on the evaluation set.

        Args:
            use_test (bool): Determine whether to use the test set, defaults to false

        Returns:
            float: The average loss on the evaluation set.
        """
        total_loss = 0
        
        #TODO add loss function to input parameters
        
        if use_test and self.test_loader is None:
            LOGGER.warning(
                "Test loader is not available. Evaluating on evaluation set instead."
            )
            loader = self.eval_loader
        elif use_test and self.test_loader is not None:
            loader = self.test_loader
        else:
            loader = self.eval_loader
        
        with torch.no_grad():
            for batch in loader:
                #TODO get the corresponding inputs(x) and labels(y)
                #TODO calculate outputs
                #TODO calculate loss with loss function
                #TODO add the loss to the total loss
                ...
        return total_loss / len(loader)
        
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
    
    def save_model(self, path: str | Path):
        """Save the model
        
        Args:
            path (str): The path to save the model to
        """
        self.model.save_model(path)
        LOGGER.info(f"Model saved to {path}")
        
    def load_model(self, path: str | Path):
        """Load the model
        
        Args:
            path (str): The path to load the model from
        """
        self.model.load(path)
        LOGGER.info(f"Model loaded from {path}")