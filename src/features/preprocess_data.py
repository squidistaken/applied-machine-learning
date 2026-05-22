import argparse
from src.constants import LOGGER
from src.features.preprocess_pytorch import (
    preprocess_data as preprocess_pytorch_data,
)
from src.features.preprocess_lightgbm import (
    preprocess_data as preprocess_lightgbm_data,
)


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(description="Run preprocessing pipelines.")

    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["all", "pytorch", "lightgbm"],
        default="all",
        help="Which preprocessing pipeline to run. Defaults to 'all'.",
    )

    parser.add_argument(
        "--lgb-size",
        type=int,
        default=64,
        help="Edge size for downsampling in LightGBM feature extraction. Defaults to 64.",
    )

    args = parser.parse_args()

    if args.pipeline in ["all", "pytorch"]:
        LOGGER.info("Starting PyTorch Preprocessing Pipeline.\n")
        preprocess_pytorch_data()
        LOGGER.info("PyTorch Preprocessing Pipeline Finished.\n")

    if args.pipeline in ["all", "lightgbm"]:
        LOGGER.info("Starting LightGBM Preprocessing Pipeline.\n")
        target_size = (args.lgb_size, args.lgb_size)
        preprocess_lightgbm_data(target_size=target_size)
        LOGGER.info("LightGBM Preprocessing Pipeline Finished.")


if __name__ == "__main__":
    main()
