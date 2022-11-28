import random

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Fix seeds for python random, numpy and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



