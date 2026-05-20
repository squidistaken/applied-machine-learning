import argparse
import itertools
import copy
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
from src.constants import LOGGER, DEVICE, RESULTS_DIR
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.models.cnn import CNN
from src.models.resnet import ResNet
from src.models.lgbm import LightGBM
from src.training.trainer import Trainer
from typing import Union, Optional

# Fallbacks.
DEFAULT_PYTORCH_LR = 1e-3
DEFAULT_LGBM_LR = 0.05
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_LEAVES = 31
DEFAULT_MAX_DEPTH = -1


def cross_validate(
    model_class: type,
    dataset_train: Union[ChestXRayDatasetPyTorch, ChestXRayDatasetLightGBM],
    dataset_val: Optional[ChestXRayDatasetPyTorch] = None,
    n_splits: int = 5,
    epochs: int = 10,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: Optional[float] = None,
    weight_decay: float = 0.0,
    device: str = DEVICE,
    num_leaves: int = DEFAULT_NUM_LEAVES,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> dict[str, float]:
    """
    Run stratified k-fold cross-validation.

    Args:
        model_class (type): The model class to instantiate.
        dataset_train (Union[ChestXRayDatasetPyTorch, ChestXRayDatasetLightGBM]): The training/base dataset.
        dataset_val (Optional[ChestXRayDatasetPyTorch]): The validation
                                                         dataset. Defaults to
                                                         None.
        n_splits (int): The number of folds. Defaults to 5.
        epochs (int): The number of epochs/boosting rounds to train per fold.
                      Defaults to 10.
        batch_size (int): The size of the batches. Defaults to
                          DEFAULT_BATCH_SIZE.
        learning_rate (float, optional): The learning rate. Defaults to None.
        weight_decay (float): The weight decay (L2 penalty) for PyTorch
                              models. Defaults to 0.0.
        device (str): The device to run models on. Defaults to DEVICE.
        num_leaves (int): The number of leaves (LightGBM only). Defaults to
                          DEFAULT_NUM_LEAVES.
        max_depth (int): The maximum tree depth (LightGBM only). Defaults to
                         DEFAULT_MAX_DEPTH.

    Returns:
        dict[str, float]: A dictionary containing average metrics across all
                          folds.
    """
    is_pytorch = model_class in [CNN, ResNet]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {"macro_f1": [], "precision": [], "recall": []}

    if is_pytorch:
        if not isinstance(dataset_train, ChestXRayDatasetPyTorch):
            raise TypeError(
                f"Expected ChestXRayDatasetPyTorch for PyTorch models, but got {type(dataset_train)}"
            )
        dataset_pytorch = dataset_train

        if dataset_val is None:
            raise ValueError(
                "dataset_val must be provided for PyTorch cross-validation to prevent data leakage."
            )

        dataset_val_pytorch = dataset_val

        lr = learning_rate if learning_rate is not None else DEFAULT_PYTORCH_LR

        # By extracting labels, we get stratified splitting.
        labels = [target for _, target in dataset_pytorch.samples]
        splits = skf.split(np.zeros(len(labels)), labels)
    else:
        if not isinstance(dataset_train, ChestXRayDatasetLightGBM):
            raise TypeError(
                f"Expected ChestXRayDatasetLightGBM for LightGBM models, but got {type(dataset_train)}"
            )
        dataset_lgbm = dataset_train
        lr = learning_rate if learning_rate is not None else DEFAULT_LGBM_LR
        X, y = dataset_lgbm.get_data()
        splits = skf.split(X, y)

    for fold, (train_idx, val_idx) in enumerate(splits):
        LOGGER.info(f"Starting Fold {fold + 1}/{n_splits}...")

        if is_pytorch:
            train_subset = Subset(dataset_pytorch, train_idx)
            val_subset = Subset(dataset_val_pytorch, val_idx)

            model = model_class(
                dataset=dataset_pytorch,
                learning_rate=lr,
                weight_decay=weight_decay,
            )

            trainer = Trainer(
                model=model,
                train_data=train_subset,
                eval_data=val_subset,
                batch_size=batch_size,
                device=device,
            )

            trainer.train(num_epochs=epochs, learning_rate=lr, patience=3)
            metrics = trainer.evaluate()
        else:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = LightGBM(
                dataset=dataset_lgbm,
                learning_rate=lr,
                num_leaves=num_leaves,
                max_depth=max_depth,
            )

            model.backward_pass(
                x_train=X_train,
                y_train=y_train,
                x_val=X_val,
                y_val=y_val,
                num_boost_round=epochs,
                patience=5,
            )

            metrics = model.evaluate(x_test=X_val, y_test=y_val)

        for key in fold_metrics.keys():
            fold_metrics[key].append(metrics[key])

        LOGGER.info(
            f"Fold {fold + 1} Metrics: Macro-F1: {metrics['macro_f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
        )

    avg_metrics = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
    return avg_metrics


def run_grid_search(
    model_name: str, splits: int, epochs: int, device: str
) -> None:
    """
    Run a grid search over cross-validation configurations.
    """
    LOGGER.info(
        f"\n{'=' * 60}\nInitialising Grid Search CV (Folds: {splits}) for {model_name.upper()}."
    )

    # NOTE: This is where you can configure the grid.
    if model_name in ["cnn", "resnet"]:
        grid = {
            "lr": [1e-4, 1e-3, 1e-2],
            "batch_size": [16, 32],
            "weight_decay": [0.0, 1e-4],
        }
    else:
        grid = {
            "lr": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [4, 6, -1],
        }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    LOGGER.info(
        f"Found {len(combinations)} parameter configurations to evaluate."
    )

    if model_name in ["cnn", "resnet"]:
        base_dataset = ChestXRayDatasetPyTorch(split="train", augment=False)
        dataset_train = copy.copy(base_dataset)
        dataset_train.transform = ChestXRayDatasetPyTorch.compose_transforms(
            augment=True
        )
        dataset_val = base_dataset
        model_class = CNN if model_name == "cnn" else ResNet
    else:
        dataset_train = ChestXRayDatasetLightGBM(split="train", augmented=False)
        model_class = LightGBM

    results = []
    best_score = -1.0
    best_config = None

    for i, config in enumerate(combinations):
        LOGGER.info(
            f"Evaluating Configuration {i + 1}/{len(combinations)}: {config}\n"
        )

        if model_name in ["cnn", "resnet"]:
            metrics = cross_validate(
                model_class=model_class,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                n_splits=splits,
                epochs=epochs,
                batch_size=config["batch_size"],
                learning_rate=config["lr"],
                weight_decay=config["weight_decay"],
                device=device,
            )
        else:
            metrics = cross_validate(
                model_class=model_class,
                dataset_train=dataset_train,
                n_splits=splits,
                epochs=epochs,
                learning_rate=config["lr"],
                num_leaves=config["num_leaves"],
                max_depth=config["max_depth"],
            )

        score = metrics["macro_f1"]
        results.append((config, metrics))

        LOGGER.info(f"Configuration {i + 1} average Macro-F1: {score:.4f}")

        if score > best_score:
            best_score = score
            best_config = config

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Grid Search Summary:\n")
    for idx, (conf, metr) in enumerate(results):
        LOGGER.info(
            f"[{idx + 1:02d}] Config: {conf} --> Macro-F1: {metr['macro_f1']:.4f} | Precision: {metr['precision']:.4f} | Recall: {metr['recall']:.4f}"
        )

    LOGGER.info(f"Best configuration found: {best_config}")
    LOGGER.info(f"Best Macro-F1: {best_score:.4f}\n")

    grid_report_path = RESULTS_DIR / f"{model_name}_grid_search_results.txt"
    with open(grid_report_path, "w") as f:
        f.write("==================================================\n")
        f.write(" {model_name.upper()} GRID SEARCH CV RESULTS REPORT\n")
        f.write("==================================================\n\n")
        f.write("CONFIGURATION SUMMARY:\n")
        f.write("Folds (K) : {splits}\n")
        f.write("Epochs    : {epochs}\n")
        f.write("Total Combinations Tested: {len(combinations)}\n\n")
        f.write("DETAILED RUN LOG:\n")
        f.write("-----------------\n")
        for idx, (conf, metr) in enumerate(results):
            f.write(f"[{idx + 1:02d}] Configuration: {conf}\n")
            f.write(f"     -> Macro-F1 : {metr['macro_f1']:.6f}\n")
            f.write(f"     -> Precision: {metr['precision']:.6f}\n")
            f.write(f"     -> Recall   : {metr['recall']:.6f}\n\n")

        f.write("\n" + "*" * 60 + "\n")
        f.write("BEST CONFIGURATION RETRIEVED:\n")
        f.write(f"Configuration : {best_config}\n")
        f.write(f"Best Macro-F1 : {best_score:.6f}\n")
        f.write("*" * 60 + "\n")
    LOGGER.info(f"Saved Grid Search report to: {grid_report_path}")


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Run Stratified K-Fold Cross Validation."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet", "lgbm"],
        required=True,
        help="The model to cross-validate.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of folds (k). Defaults to 5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs/boost rounds. Defaults dynamically based on the chosen model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Defaults dynamically (1e-3 for PyTorch, 0.05 for LightGBM).",
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
        help=f"Device for PyTorch models (default: {DEVICE}).",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable hyperparameter grid search cross-validation.",
    )

    args = parser.parse_args()

    # Epochs and learning rates vary depending on the type of model, so these
    # are "fallbacks."
    epochs = args.epochs
    if epochs is None:
        if args.model == "cnn":
            epochs = 20
        elif args.model == "resnet":
            epochs = 10
        elif args.model == "lgbm":
            epochs = 150

    lr = args.lr
    if lr is None:
        lr = (
            DEFAULT_PYTORCH_LR
            if args.model in ["cnn", "resnet"]
            else DEFAULT_LGBM_LR
        )

    if args.grid_search:
        run_grid_search(
            model_name=args.model,
            splits=args.splits,
            epochs=epochs,
            device=args.device,
        )
    else:
        LOGGER.info(
            f"Initialising standard cross validation for {args.model.upper()} with {args.splits} folds."
        )
        if args.model in ["cnn", "resnet"]:
            base_dataset = ChestXRayDatasetPyTorch(split="train", augment=False)
            dataset_train = copy.copy(base_dataset)
            dataset_train.transform = (
                ChestXRayDatasetPyTorch.compose_transforms(augment=True)
            )
            dataset_val = base_dataset
            model_class = CNN if args.model == "cnn" else ResNet

            avg_metrics = cross_validate(
                model_class=model_class,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                n_splits=args.splits,
                epochs=epochs,
                batch_size=DEFAULT_BATCH_SIZE,
                learning_rate=lr,
                weight_decay=args.weight_decay,
                device=args.device,
            )
        elif args.model == "lgbm":
            dataset_train = ChestXRayDatasetLightGBM(
                split="train", augmented=False
            )

            avg_metrics = cross_validate(
                model_class=LightGBM,
                dataset_train=dataset_train,
                n_splits=args.splits,
                epochs=epochs,
                learning_rate=lr,
                num_leaves=DEFAULT_NUM_LEAVES,
                max_depth=DEFAULT_MAX_DEPTH,
            )

        # Save standard non-grid search CV results report as a text file
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cv_report_path = RESULTS_DIR / f"{args.model}_cv_results.txt"
        with open(cv_report_path, "w") as f:
            f.write("==================================================\n")
            f.write(f" {args.model.upper()} STANDARD CV PERFORMANCE REPORT\n")
            f.write("==================================================\n\n")
            f.write("CONFIGURATIONS:\n")
            f.write("---------------\n")
            f.write(f"Folds (K)                  : {args.splits}\n")
            f.write(f"Epochs / Boosting Rounds   : {epochs}\n")
            f.write(f"Defaults - LR              : {lr}\n")
            if args.model in ["cnn", "resnet"]:
                f.write(f"Defaults - Batch Size      : {DEFAULT_BATCH_SIZE}\n")
                f.write(f"Defaults - Weight Decay    : {args.weight_decay}\n")
                f.write(f"Device                     : {args.device}\n")
            elif args.model == "lgbm":
                f.write(f"Defaults - Num Leaves      : {DEFAULT_NUM_LEAVES}\n")
                f.write(f"Defaults - Max Depth       : {DEFAULT_MAX_DEPTH}\n")
            f.write("\n")

            f.write("CROSS VALIDATION PERFORMANCE (AVERAGE ACROSS FOLDS):\n")
            f.write("----------------------------------------------------\n")
            for k, v in avg_metrics.items():
                f.write(f"Average {k:<12} : {v:.6f}\n")
        LOGGER.info(f"Saved cross-validation results to: {cv_report_path}")


if __name__ == "__main__":
    main()
