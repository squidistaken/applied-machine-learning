import argparse
import copy
from typing import Callable, Literal, Any, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from src.constants import LOGGER, DEVICE, RESULTS_DIR, MODELS_DIR
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.models.cnn import CNN
from src.models.resnet import ResNet
from src.models.lgbm import LightGBM
from src.training.trainer import Trainer
import json


def train_model(
    model_name: Literal["cnn", "resnet", "lgbm"],
    epochs: int,
    lr: float,
    batch_size: int = 32,
    patience: int = 3,
    num_leaves: int = 31,
    max_depth: int = -1,
    weight_decay: float = 0.0,
    device: str = DEVICE,
    enable_uq: bool = True,
    on_epoch_end: Optional[Callable[[int, int, dict], None]] = None,
) -> None:
    """
    Train a model.

    Args:
        model_name (Literal["cnn", "resnet", "lgbm"]): The model train.
        epochs (int): The number of epochs.
        lr (float): The learning rate.
        batch_size (int): The batch size (PyTorch only). Defaults to 32.
        patience (int): The patience for early stopping. Defaults to 3.
        num_leaves (int): The number of leaves (LightGBM only). Defaults to 31.
        max_depth (int): The maximum tree depth (LightGBM only). Defaults to
                         -1.
        weight_decay (float): The weight decay (L2 penalty) (PyTorch only).
                              Defaults to 0.0.
        device (str): The device to run models on. Defaults to DEVICE.
        enable_uq (bool): Whether to calculate calibration and Predictive Entropy metrics.
        on_epoch_end (Optional[Callable[[int, int, dict], None]]): Called as
            ``(current_epoch, total_epochs, metrics)`` at the end of each epoch
            for live progress reporting.
    """
    LOGGER.info(
        f"Initializing training run for {model_name.upper()} on device: {device} (UQ Enabled: {enable_uq})..."
    )

    if model_name in ["cnn", "resnet"]:
        base_dataset = ChestXRayDatasetPyTorch(split="train", augment=False)
        test_data = ChestXRayDatasetPyTorch(split="test", augment=False)
        train_dataset = copy.copy(base_dataset)
        train_dataset.transform = ChestXRayDatasetPyTorch.compose_transforms(
            augment=True
        )
        val_dataset = base_dataset

        labels = [target for _, target in base_dataset.samples]

        # NOTE: 80/20 train-val split.
        train_idx, val_idx = train_test_split(
            range(len(labels)), test_size=0.2, stratify=labels, random_state=42
        )

        train_data = Subset(train_dataset, train_idx)
        eval_data = Subset(val_dataset, val_idx)

        if model_name == "cnn":
            model = CNN(
                dataset=train_dataset,
                learning_rate=lr,
                weight_decay=weight_decay,
            )
        else:
            model = ResNet(
                dataset=train_dataset,
                learning_rate=lr,
                weight_decay=weight_decay,
            )

    elif model_name == "lgbm":
        full_train = ChestXRayDatasetLightGBM(split="train", augmented=False)
        test_data = ChestXRayDatasetLightGBM(split="test", augmented=False)

        # NOTE: 80/20 train-val split.
        X_train_df, X_val_df, y_train_ser, y_val_ser = train_test_split(
            full_train.X,
            full_train.y,
            test_size=0.2,
            stratify=full_train.y,
            random_state=42,
        )
        train_data = copy.copy(full_train)
        train_data.X = X_train_df
        train_data.y = y_train_ser
        eval_data = copy.copy(full_train)
        eval_data.X = X_val_df
        eval_data.y = y_val_ser
        model = LightGBM(
            dataset=train_data,
            learning_rate=lr,
            num_leaves=num_leaves,
            max_depth=max_depth,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    trainer = Trainer(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        device=device,
        enable_uq=enable_uq,
    )

    LOGGER.info("Starting training...")
    trainer.train(
        num_epochs=epochs,
        learning_rate=lr,
        patience=patience,
        epoch_callback=on_epoch_end,
    )
    final_epoch = len(trainer.history["train_loss"])
    LOGGER.info("Evaluating validation dataset performance...")
    val_metrics = trainer.evaluate(use_test=False)
    LOGGER.info(f"Validation Metrics: {val_metrics}")
    LOGGER.info("Evaluating final independent test dataset performance...")
    test_metrics = trainer.evaluate(use_test=True)
    LOGGER.info(f"Final Test Metrics: {test_metrics}")
    trainer.model.save_model()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.plot_history(show=False)
    trainer.plot_confusion_matrix(show=False, use_test=True)

    calibration_report = None

    if enable_uq:
        scaler = trainer.fit_temperature_scaler()
        temp_path = MODELS_DIR / f"{model.__class__.__name__}_temperature.json"
        scaler.save(temp_path)
        LOGGER.info(f"Saved temperature scaler to: {temp_path}")

        trainer.plot_reliability_diagram(use_test=True, calibrator=scaler)
        trainer.plot_selective_prediction(use_test=True)

        calibration_report = trainer.uq_report(use_test=True, calibrator=scaler)
        LOGGER.info(f"Calibration report (test): {calibration_report}")

    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"

    hyperparameters: dict[str, Any] = {
        "learning_rate": lr,
        "epochs": epochs,
        "final_epoch": final_epoch,
        "patience": patience,
        "enable_uq": enable_uq,
    }

    if model_name in ["cnn", "resnet"]:
        hyperparameters.update(
            {
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "device": device,
            }
        )
    elif model_name == "lgbm":
        hyperparameters.update(
            {
                "num_leaves": num_leaves,
                "max_depth": max_depth,
            }
        )

    metrics_data = {
        "hyperparameters": hyperparameters,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    if calibration_report is not None:
        metrics_data["calibration"] = calibration_report

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    LOGGER.info(f"Saved evaluation JSON report to: {metrics_path}")


def main() -> None:
    """Run the training script from the command line."""
    parser = argparse.ArgumentParser(
        description="Train a machine learning model for chest X-ray classification."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet", "lgbm"],
        required=True,
        help="The model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (or boost rounds for LightGBM). Defaults dynamically based on the chosen model.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (PyTorch only)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Defaults dynamically (1e-3 for PyTorch, 0.05 for LightGBM).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs to wait for improvement before early stopping (PyTorch & LightGBM).",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="Number of leaves (LightGBM only).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
        help="Maximum tree depth (LightGBM only).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 penalty) for PyTorch models. Defaults to 0.0.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to run PyTorch models on (default: {DEVICE}).",
    )
    parser.add_argument(
        "--no-uq",
        dest="enable_uq",
        action="store_false",
        help="Disable slower Uncertainty Quantification metrics during validation loops for training speedup.",
    )
    parser.set_defaults(enable_uq=True)

    args = parser.parse_args()

    epochs = args.epochs
    if epochs is None:
        if args.model == "cnn":
            epochs = 20
        elif args.model == "resnet":
            epochs = 10
        elif args.model == "lgbm":
            epochs = 150
        else:
            epochs = 20

    lr = args.lr
    if lr is None:
        lr = 1e-3 if args.model in ["cnn", "resnet"] else 0.05

    train_model(
        model_name=args.model,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=lr,
        patience=args.patience,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        weight_decay=args.weight_decay,
        device=args.device,
        enable_uq=args.enable_uq,
    )


if __name__ == "__main__":
    main()
