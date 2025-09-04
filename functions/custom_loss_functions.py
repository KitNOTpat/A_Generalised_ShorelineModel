# custom_loss_functions

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#####################################################
#####################################################

def quantile_loss(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the quantile loss for a TFT model output across all samples, horizons, and fixed quantiles (0.11, 0.5, 0.89) 
    - Shoutout to Richard McElreath (Statistical Rethinking).
    
    ignoring places where targets == -9999.

    Parameters
    ----------
    output : torch.Tensor
        Model outputs with shape [batch_size, forecast_horizon, 3] (3 quantiles: 0.11, 0.5, 0.89).
    targets : torch.Tensor
        Ground truth values with shape [batch_size, forecast_horizon].

    Returns
    -------
    q_loss : torch.Tensor
        A scalar tensor representing the average quantile loss across all samples, horizons, and quantiles.
    """

    if torch.all(targets == -9999):
        return torch.tensor(0.0, device=output.device, requires_grad=True)
    
    # Fixed quantiles
    quantiles = torch.tensor([0.055, 0.5, 0.945], device=output.device)  # [3]

    # Create mask for valid target values (ignore where targets == 0)
    mask = targets != -9999  # [batch_size, forecast_horizon]

    mask_expanded = mask.unsqueeze(-1).expand_as(output)  # [batch_size, forecast_horizon, 3]

    # Align targets shape with outputs for quantile-wise operations
    targets = targets.unsqueeze(-1)  # [batch_size, forecast_horizon, 1]

    # Compute errors for quantile loss (pinball loss)

    errors = targets - output  # [batch_size, forecast_horizon, 3]

    # Apply the quantile loss formula
    losses = torch.maximum(
        (quantiles - 1) * errors,  # Under-prediction penalty
        quantiles * errors         # Over-prediction penalty
    )  # [batch_size, forecast_horizon, 3]

    # Apply the mask correctly
    masked_losses = torch.where(mask_expanded, losses, torch.tensor(0.0, device=output.device))

    # Compute the mean, only considering valid entries
    valid_count = mask_expanded.sum()  # Number of valid (non-zero) elements
    q_loss = masked_losses.sum() / valid_count  # Properly normalized loss

    return q_loss