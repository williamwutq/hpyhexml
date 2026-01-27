# Reference Materials


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hpyhex import Hex, HexEngine


def get_list_of_correspondence_lists(
    engine_radius: int,
    kernel_radius: int,
) -> np.ndarray:
    """Generates a list of correspondence lists for hexagonal convolution.

    Args:
    engine_radius (int): Radius of the hexagonal engine.
    kernel_radius (int): Radius of the convolution kernel.

    Returns:
    np.ndarray: 2D array of correspondence lists for each position in the kernel.
    """
    if kernel_radius == 0:
        return np.array([[]], dtype=np.int64)  # Empty kernel
    elif kernel_radius == 1:
        return np.array([
            HexEngine.to_numpy_correspondence_list_int64(engine_radius, Hex())
        ]) # Identity
    else:
        length = HexEngine.solve_length(kernel_radius)
        correspondence_lists = [
            HexEngine.to_numpy_correspondence_list_int64(
            engine_radius,
                HexEngine.hpyhex_rs_coordinate_block(kernel_radius, i)
                .shift_i(1-kernel_radius)
                .shift_k(1-kernel_radius)
                # This is to center the kernel at (0,0) because this version of HexEngine has origin at left most corner
            )
            for i in range(length)
        ]
        return np.array(correspondence_lists)


def conv_by_list_of_correspondence_lists(
    x: torch.Tensor,
    weight: torch.Tensor,
    correspondence_lists: np.ndarray,
) -> torch.Tensor:
    """Performs hexagonal convolution using correspondence lists.

    Args:
    x (torch.Tensor): Input tensor of shape (B, C_in, L).
    weight (torch.Tensor): Weight tensor of shape (C_out, C_in, K).
    correspondence_lists (np.ndarray): 2D array of correspondence lists.

    Returns:
    torch.Tensor: Output tensor after convolution.
    """
    B, C_in, L = x.shape
    C_out, _, K = weight.shape
    device = x.device

    # Prepare output tensor
    out = torch.zeros((B, C_out, L), device=device)

    # correspondence_lists may contain -1 for invalid mappings

    # Prepare an index tensor for advanced indexing
    # Shape: (K, L)
    corr = torch.from_numpy(correspondence_lists).to(x.device)  # (K, L)

    # Mask for valid indices
    valid_mask = corr != -1  # (K, L)

    # Replace -1 with 0 for safe indexing (will be masked out later)
    corr_safe = corr.clone()
    corr_safe[~valid_mask] = 0

    # Gather input values for all kernel positions and all spatial positions
    # x: (B, C_in, L)
    # We want to gather along the last dim (L) using corr_safe (K, L)
    # Result: (B, C_in, K, L)
    x_gathered = x.unsqueeze(2).expand(-1, -1, K, -1)  # (B, C_in, K, L)
    corr_safe_expanded = corr_safe.unsqueeze(0).unsqueeze(0).expand(B, C_in, -1, -1)  # (B, C_in, K, L)
    x_vals = torch.gather(x_gathered, 3, corr_safe_expanded)  # (B, C_in, K, L)

    # Zero out invalid positions
    valid_mask_expanded = valid_mask.unsqueeze(0).unsqueeze(0).expand(B, C_in, -1, -1)
    x_vals = x_vals * valid_mask_expanded

    # weight: (C_out, C_in, K)
    # We want to multiply x_vals (B, C_in, K, L) by weight (C_out, C_in, K) and sum over C_in and K
    # Reshape weight for broadcasting: (1, C_out, C_in, K, 1)
    weight_exp = weight.unsqueeze(0).unsqueeze(-1)  # (1, C_out, C_in, K, 1)
    x_vals_exp = x_vals.unsqueeze(1)  # (B, 1, C_in, K, L)

    # Multiply and sum over C_in and K
    out = (x_vals_exp * weight_exp).sum(dim=2).sum(dim=2)  # (B, C_out, L)

    return out