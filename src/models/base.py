from src.data.dataset import ChestXRayDataset
class BaseModel:
    def __init__(self, dataset : ChestXRayDataset):
        # TODO import the dataset and extract the important attributes
        self.dataset = dataset
        self.classes = self.dataset.classes

    def forward_pass(self, x):
        """
        predict class of input

        Args:
            x : input data (image)
        """
        ...

    def backward_pass(self, x_train, y_train, **kwargs):
        """
        train model

        Args:
            x : input data (image)
            y : class label(s)
        """
        ...

    def evaluate(self, x_test, y_test):
        # TODO return different hyperparameters for evaluation or
        #   make separate classes for each hyperparameter?
        """
        test the performance of the model

        Args:
            x_test : input data (image)
            y_test : class label(s)
        """
        ...

    def save(self, path):
        """
        Save model weights
        """
        ...

    def load(self, path):
        """
        Load a trained model
        """
        ...
