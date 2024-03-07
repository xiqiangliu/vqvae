import logging

import torch

_logger = logging.getLogger(__name__)
_logger.info(f"Using torch version {torch.__version__}")

if torch.cuda.is_available():
    _logger.info(f"Using {torch.cuda.get_device_name()} for acceleration")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    _logger.info("Using MPS for acceleration")
    device = torch.device("mps")
else:
    _logger.info("Using CPU for acceleration")
    device = torch.device("cpu")

__all__ = ["device"]
