from pathlib import Path
from src.utils.logger import Logger
import yaml

ROOT_DIR = Path(__file__).parent.parent

with open(ROOT_DIR / "config.yaml", "r") as config_file:
    data = yaml.load(config_file, Loader=yaml.SafeLoader)
    paths = data["paths"]

DATA_DIR = ROOT_DIR / Path(paths["data"])
MODEL_DIR = ROOT_DIR / Path(paths["model"])
LOGS_DIR = ROOT_DIR / Path(paths["logs"])
DEBUG = data["debug"]
LOGGER = Logger("aml")
