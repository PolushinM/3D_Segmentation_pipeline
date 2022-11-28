from math import log

import torch


def dice_coef(targets: torch.tensor, inputs: torch.tensor, threshold: float = 0.5) -> float:
    """Dice coefficient metric with variable confidence threshold, corresponding to sigmoid.
        Note: if you need metric function of the form mertic_func(targets: torch.tensor, inputs: torch.tensor)
        with non-default threshold, use, for example, lambda function: "lambda x, y: dice_coef(x, y, 0.2)".
                    Parameters
                    ----------
                    targets : {torch.tensor} ground truth segmentation mask batch (values from set {0, 1}).
                    inputs : {torch.tensor} predicted segmentation mask batch (values from range [0, 1]).
                        ("logits", outputs without sigmoid activation).
                    threshold : {float} confidence threshold for mask calculation, default 0.5.

                    Returns
                    -------
                    dice : {float} Dice coefficient.
                    """

    inputs = (inputs > -log(1 / threshold + 1)) * 1.0

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).mean()

    dice = 2. * intersection / (inputs.mean() + targets.mean())
    return dice.item()


def dice_loss(targets: torch.tensor, inputs: torch.tensor, smooth=1) -> torch.Tensor:
    """Dice loss function.
                    Parameters
                    ----------
                    targets : {torch.tensor} ground truth segmentation mask batch (values from set {0, 1}).
                    inputs : {torch.tensor} predicted segmentation mask batch (values from range [0, 1]).
                        ("logits", outputs without sigmoid activation).
                    smooth : {str} smoothing coefficient, default 1.0.

                    Returns
                    -------
                    dice : {torch.Tensor} Loss value.
                    """
    inputs = torch.sigmoid(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice
