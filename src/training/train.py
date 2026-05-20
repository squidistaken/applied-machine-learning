# TODO: An all-purpose training loop for models.
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_model(model, train_dataset, val_dataset=None, epochs=10, batch_size=32, learning_rate=1e-3, device=None,):
    """
    this function trains the PyTorch model on the chest X-ray dataset.

    Args:
        model: PyTorch model to train.
        train_dataset: training dataset.
        val_dataset: validation dataset, optional.
        epochs: number of training epochs.
        batch_size: batch size.
        learning_rate: learning rate for optimizer.
        device: "cuda", "mps", or "cpu".

    Returns:
        model: trained model.
        history: dictionary containing training and validation losses.
    """

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Training loss: {avg_train_loss:.4f}")

        if val_loader is not None:
            val_loss, val_accuracy = evaluate_model(
                model,
                val_loader,
                criterion,
                device
            )

            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            print(f"Validation loss: {val_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")

        print("-" * 40)

    return model, history


def evaluate_model(model, data_loader, criterion, device):
    """
    this is meant to evaluate the model (during the training, so validation (?))
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def save_model(model, path):
    """
    here we save the trained model weights.
    """

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
