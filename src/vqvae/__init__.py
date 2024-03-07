from pathlib import Path

from .utils.logging import logger as root_logger  # noqa: F401 # isort:skip
from .utils.device import device  # noqa: F401

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
