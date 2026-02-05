# Hexagonal convolution layers using correspondence lists for PyTorch.
'''
Provides hexagonal convolution layers that operate directly on hexagonal grids
using correspondence lists for memory efficiency. Includes standard hexagonal
convolution, custom kernel convolution, shrinking, and moving layers.

Kernels:
- PureHexConv: Standard hexagonal convolution with radius-based kernels. Trainable.
- MaskedHexConv: Hexagonal convolution with mask modulation. Trainable.
- CustomHexConv: Hexagonal convolution with user-defined kernel shapes. Trainable.
- HexShrink: Shrinks the hexagonal grid by removing outer layers. Not trainable.
- HexMove: Moves the hexagonal grid by a specified Hex offset. Not trainable.
- HexDense: Dense layer for hexagonal grids. Trainable.
- HexBias: Bias layer for hexagonal grids. Trainable.
- HexMerge: Bilinear merge layer for hexagonal grids. Trainable.
'''

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import hpyhex
    from hpyhex import Hex, HexEngine
except ImportError:
    raise ImportError(
        """
        hpyhex-rs module is required for optimized hexagonal convolutions.
        Please install it with `pip install hpyhex-rs` and uninstall
        the `hpyhex` package if previously installed.
        """
    )


def get_list_of_correspondence_lists(
    engine_radius: int,
    kernel_radius: int,
) -> np.ndarray:
    """
    Generates a list of correspondence lists for hexagonal convolution.

    It's best to reuse the output of this function rather than
    recomputing it every forward pass for efficiency.

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


def get_list_of_custom_correspondence_lists(
    engine_radius: int,
    custom_positions: list[Hex],
) -> np.ndarray:
    """
    Generates a list of correspondence lists for custom hexagonal convolution.

    It's best to reuse the output of this function rather than
    recomputing it every forward pass for efficiency.

    Args:
    engine_radius (int): Radius of the hexagonal engine.
    custom_positions (list[Hex]): List of Hex positions defining the kernel.

    Returns:
    np.ndarray: 2D array of correspondence lists for each position in the kernel.
    """
    correspondence_lists = [
        HexEngine.to_numpy_correspondence_list_int64(
            engine_radius,
            pos
        )
        for pos in custom_positions
    ]
    return np.array(correspondence_lists)



class PureHexConv(nn.Module):
    """
    Pure hexagonal convolution layer using correspondence lists.
    
    This layer applies convolution on a hexagonal grid without additional data,
    operating directly on board states. Uses correspondence lists for memory
    efficiency (46x less memory than matrix-based approach).
    
    Args:
        engine_radius (int): Radius of the hexagonal grid
        kernel_radius (int): Radius of the convolution kernel, must be >=0
                            1: self only (identity)
                            2: self + ring-1 (7 positions)  
                            3: self + ring-1 + ring-2 (19 positions)
        in_channels (int): Number of input features per cell
        out_channels (int): Number of output features per cell
        bias (bool): If True, adds a learnable bias
        
    Shape:
        - Input: (batch_size, in_channels * num_cells) or (batch_size, num_cells) if in_channels=1
        - Output: (batch_size, out_channels * num_cells) or (batch_size, num_cells) if out_channels=1
        
    Examples:
        >>> # Single-channel board state → features
        >>> conv = PureHexConv(engine_radius=5, kernel_radius=1, 
        ...                    in_channels=1, out_channels=16)
        >>> board = torch.rand(8, 61)  # batch=8, radius=5 has 91 cells
        >>> features = conv(board)     # (8, 61*16)
        
        >>> # Multi-channel features → board score
        >>> conv2 = PureHexConv(engine_radius=5, kernel_radius=2,
        ...                     in_channels=16, out_channels=1)
        >>> scores = conv2(features)   # (8, 61)
        
        >>> # Stack for deep network
        >>> model = nn.Sequential(
        ...     PureHexConv(5, 1, 1, 32),
        ...     nn.ReLU(),
        ...     PureHexConv(5, 2, 32, 64),
        ...     nn.ReLU(),
        ...     PureHexConv(5, 1, 64, 1)
        ... )
    """
    
    def __init__(
        self,
        engine_radius: int,
        kernel_radius: int,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True
    ):
        super(PureHexConv, self).__init__()

        # Not zero check
        if engine_radius <= 0:
            raise ValueError("engine_radius must be positive.")
        if kernel_radius < 0:
            raise ValueError("kernel_radius must be non-negative.")
        
        self.engine_radius = engine_radius
        self.kernel_radius = kernel_radius
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate grid size and kernel size
        self.num_cells = HexEngine.solve_length(engine_radius)
        self.kernel_size = HexEngine.solve_length(kernel_radius)
        
        # Get correspondence lists (reused, not recomputed)
        correspondence_lists = get_list_of_correspondence_lists(
            engine_radius, kernel_radius
        )
                
        # Precompute valid mask
        valid_mask = correspondence_lists != -1
        self.register_buffer(
            'valid_mask',
            torch.from_numpy(valid_mask).float()  # (kernel_size, num_cells)
        )

        # Compute safe correspondence lists
        corr_safe = correspondence_lists.copy()
        corr_safe[~valid_mask] = 0  # Replace -1 with 0
        
        # Register as buffer (not a parameter, moves with model to device)
        self.register_buffer(
            'corr_safe',
            torch.from_numpy(corr_safe).long()  # (kernel_size, num_cells)
        )
        
        # Learnable parameters
        # Weight: (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.kernel_size)
        )
        
        if bias:
            # Bias: (out_channels,)
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using correspondence lists.
        
        Args:
            x: Input tensor (B, C_in * L) or (B, L) if C_in=1
            
        Returns:
            Output tensor (B, C_out * L) or (B, L) if C_out=1
        """
        B = x.shape[0]
        
        # Reshape input to (B, C_in, L)
        if self.in_channels == 1 and x.shape[1] == self.num_cells:
            x = x.unsqueeze(1)  # (B, 1, L)
        else:
            x = x.view(B, self.in_channels, self.num_cells)  # (B, C_in, L)
        
        # Apply convolution using correspondence lists
        out = self._conv_by_correspondence_lists(x)  # (B, C_out, L)
        
        # Reshape output
        if self.out_channels == 1:
            out = out.squeeze(1)  # (B, L)
        else:
            out = out.view(B, -1)  # (B, C_out * L)
        
        return out
    
    def _conv_by_correspondence_lists(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core convolution operation using correspondence lists.
        
        Args:
            x: Input tensor of shape (B, C_in, L)
            
        Returns:
            Output tensor of shape (B, C_out, L)
        """
        B, C_in, L = x.shape
        C_out, _, K = self.weight.shape
        
        # Get correspondence lists: (K, L)
        corr_safe = self.corr_safe
        valid_mask = self.valid_mask
        
        # Gather values for all kernel positions
        # Expand x to (B, C_in, K, L) then gather
        x_expanded = x.unsqueeze(2).expand(-1, -1, K, -1)  # (B, C_in, K, L)
        corr_expanded = corr_safe.unsqueeze(0).unsqueeze(0).expand(B, C_in, -1, -1)  # (B, C_in, K, L)
        
        x_vals = torch.gather(x_expanded, 3, corr_expanded)  # (B, C_in, K, L)
        
        # Zero out invalid positions
        valid_expanded = valid_mask.unsqueeze(0).unsqueeze(0).expand(B, C_in, -1, -1)
        x_vals = x_vals * valid_expanded  # (B, C_in, K, L)
        
        # Apply weights: (C_out, C_in, K) × (B, C_in, K, L) → (B, C_out, L)
        weight_exp = self.weight.unsqueeze(0).unsqueeze(-1)  # (1, C_out, C_in, K, 1)
        x_vals_exp = x_vals.unsqueeze(1)  # (B, 1, C_in, K, L)
        
        # Multiply and sum over C_in and K
        out = (x_vals_exp * weight_exp).sum(dim=2).sum(dim=2)  # (B, C_out, L)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)  # (1, C_out, 1)
        
        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'engine_radius={self.engine_radius}, '
                f'kernel_radius={self.kernel_radius}, '
                f'num_cells={self.num_cells}, '
                f'kernel_size={self.kernel_size}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'bias={self.bias is not None}')
    

class MaskedHexConv(nn.Module):
    """
    Masked hexagonal convolution layer using correspondence lists with input mask.
    
    Similar to PureHexConv but incorporates a floating-point mask tensor that modulates
    the convolution weights across kernel positions. The mask allows for dynamic
    weighting of different kernel components.
    
    Args:
        engine_radius (int): Radius of the hexagonal grid
        kernel_radius (int): Radius of the convolution kernel, must be >=0
        kernel_in (int): Number of mask input channels
        in_channels (int): Number of input features per cell
        out_channels (int): Number of output features per cell
        mask_bias (bool): If True, adds a learnable mask bias with C_in * C_out parameters
        bias (bool): If True, adds a learnable bias with C_out parameters
        
    Shape:
        - Input x: (batch_size, in_channels * num_cells) or (batch_size, num_cells) if in_channels=1
        - Input mask: (kernel_in, kernel_size)
        - Output: (batch_size, out_channels * num_cells) or (batch_size, num_cells) if out_channels=1
        
    Examples:
        >>> # Masked convolution with 3 mask channels
        >>> conv = MaskedHexConv(engine_radius=5, kernel_radius=2, 
        ...                      kernel_in=3, in_channels=1, out_channels=16)
        >>> board = torch.rand(8, 61)  # batch=8, radius=5 has 61 cells
        >>> mask = torch.rand(3, 19)   # 19 kernel positions for radius=2
        >>> features = conv(board, mask)  # (8, 61*16)
    """
    
    def __init__(
        self,
        engine_radius: int,
        kernel_radius: int,
        kernel_in: int,
        in_channels: int = 1,
        out_channels: int = 1,
        mask_bias: bool = False,
        bias: bool = True
    ):
        super(MaskedHexConv, self).__init__()

        # Validation
        if engine_radius <= 0:
            raise ValueError("engine_radius must be positive.")
        if kernel_radius < 0:
            raise ValueError("kernel_radius must be non-negative.")
        if kernel_in <= 0:
            raise ValueError("kernel_in must be positive.")
        
        self.engine_radius = engine_radius
        self.kernel_radius = kernel_radius
        self.kernel_in = kernel_in
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate grid size and kernel size
        self.num_cells = HexEngine.solve_length(engine_radius)
        self.kernel_size = HexEngine.solve_length(kernel_radius)
        
        # Get correspondence lists (reused, not recomputed)
        correspondence_lists = get_list_of_correspondence_lists(
            engine_radius, kernel_radius
        )
                
        # Precompute valid mask
        valid_mask = correspondence_lists != -1
        self.register_buffer(
            'valid_mask',
            torch.from_numpy(valid_mask).float()  # (kernel_size, num_cells)
        )

        # Compute safe correspondence lists
        corr_safe = correspondence_lists.copy()
        corr_safe[~valid_mask] = 0  # Replace -1 with 0
        
        # Register as buffer (not a parameter, moves with model to device)
        self.register_buffer(
            'corr_safe',
            torch.from_numpy(corr_safe).long()  # (kernel_size, num_cells)
        )
        
        # Learnable parameters
        # Weights: (kernel_in, in_channels, out_channels)
        self.weights = nn.Parameter(
            torch.randn(kernel_in, in_channels, out_channels)
        )
        
        if mask_bias:
            # Mask bias: (in_channels, out_channels)
            self.mask_bias = nn.Parameter(torch.zeros(in_channels, out_channels))
        else:
            self.register_parameter('mask_bias', None)
        
        if bias:
            # Bias: (out_channels,)
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        if self.mask_bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.mask_bias, -bound, bound)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using correspondence lists with mask modulation.
        
        Args:
            x: Input tensor (B, C_in * L) or (B, L) if C_in=1
            mask: Mask tensor (B, M_in, K) where K is kernel_size

        Returns:
            Output tensor (B, C_out * L) or (B, L) if C_out=1
        """
        if mask.shape[1:] != (self.kernel_in, self.kernel_size):
            raise ValueError(f"mask shape must be (B, {self.kernel_in}, {self.kernel_size}), got {mask.shape}")

        B = x.shape[0]
        
        # Reshape input to (B, C_in, L)
        if self.in_channels == 1 and x.shape[1] == self.num_cells:
            x = x.unsqueeze(1)  # (B, 1, L)
        else:
            x = x.view(B, self.in_channels, self.num_cells)  # (B, C_in, L)
        
        # Apply convolution using correspondence lists with mask
        out = self._conv_by_correspondence_lists(x, mask)  # (B, C_out, L)
        
        # Reshape output
        if self.out_channels == 1:
            out = out.squeeze(1)  # (B, L)
        else:
            out = out.reshape(B, -1)  # (B, C_out * L)
        
        return out
    
    def _conv_by_correspondence_lists(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Core masked convolution operation using correspondence lists.
        
        Args:
            x: Input tensor of shape (B, C_in, L)
            mask: Mask tensor of shape (B, kernel_in, K)

        Returns:
            Output tensor of shape (B, C_out, L)
        """
        B, C_in, L = x.shape
        _, kernel_in, K = mask.shape
        C_out = self.out_channels
        
        # Get correspondence lists: (K, L)
        corr_safe = self.corr_safe
        valid_mask = self.valid_mask
        
        # Gather values for all kernel positions
        # Expand x to (B, C_in, K, L) then gather
        x_expanded = x.unsqueeze(2).expand(-1, -1, K, -1)  # (B, C_in, K, L)
        corr_expanded = corr_safe.unsqueeze(0).unsqueeze(0).expand(B, C_in, -1, -1)  # (B, C_in, K, L)
        
        x_vals = torch.gather(x_expanded, 3, corr_expanded)  # (B, C_in, K, L)
        
        # Zero out invalid positions
        valid_expanded = valid_mask.unsqueeze(0).unsqueeze(0).expand(B, C_in, -1, -1)
        x_vals = x_vals * valid_expanded  # (B, C_in, K, L)
        
        # Apply mask modulation
        # Weights: (M_in, C_in, C_out)
        effective_weights = self.weights
        if self.mask_bias is not None:
            effective_weights = effective_weights + self.mask_bias.unsqueeze(0)  # (M_in, C_in, C_out)
        
        # Compute contributions: (B, C_in, K, L) -> (B, M_in, K, L, C_out)
        contrib = torch.einsum('bckl,mco->bmklo', x_vals, effective_weights)
        
        # Apply mask: (B, M_in, K) -> (B, M_in, K, 1, 1)
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (B, M_in, K, 1, 1)
        
        # Weight contributions
        weighted_contrib = contrib * mask_expanded  # (B, M_in, K, L, C_out)
        
        # Sum over M_in and K dimensions
        out = weighted_contrib.sum(dim=1).sum(dim=1)  # (B, L, C_out)
        
        # Transpose to (B, C_out, L)
        out = out.transpose(1, 2)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)  # (1, C_out, 1)
        
        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'engine_radius={self.engine_radius}, '
                f'kernel_radius={self.kernel_radius}, '
                f'kernel_in={self.kernel_in}, '
                f'num_cells={self.num_cells}, '
                f'kernel_size={self.kernel_size}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'mask_bias={self.mask_bias is not None}, '
                f'bias={self.bias is not None}')


class CustomHexConv(nn.Module):
    """
    Custom hexagonal convolution layer using user-defined kernel positions.
    
    Inherits from PureHexConv but allows specifying arbitrary kernel shapes
    via a list of Hex coordinates.
    
    Args:
        engine_radius (int): Radius of the hexagonal grid
        custom_positions (list[Hex]): List of Hex positions defining the kernel
        in_channels (int): Number of input features per cell
        out_channels (int): Number of output features per cell
        bias (bool): If True, adds a learnable bias

    Shape:
        - Input: (batch_size, in_channels * num_cells) or (batch_size, num_cells) if in_channels=1
        - Output: (batch_size, out_channels * num_cells) or (batch_size, num_cells) if out_channels=1
    Examples:
        >>> # Custom kernel: cross shape
        >>> custom_positions = [
        ...     Hex(0,0), Hex(1,0), Hex(-1,0),
        ...     Hex(0,1), Hex(0,-1)
        ... ]
        >>> conv = CustomHexConv(engine_radius=5,
        ...                      custom_positions=custom_positions,
        ...                      in_channels=1, out_channels=16)
        >>> board = torch.rand(8, 61)  # batch=8, radius=5 has 61 cells
        >>> features = conv(board)     # (8, 61*16)
    """

    def __init__(
        self,
        engine_radius: int,
        custom_positions: list[Hex],
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True
    ):
        super(PureHexConv, self).__init__()

        # Not zero check
        if engine_radius <= 0:
            raise ValueError("engine_radius must be positive.")
        if len(custom_positions) == 0:
            raise ValueError("custom_positions must be non-empty.")
        
        self.engine_radius = engine_radius
        self.custom_positions = custom_positions
        self.kernel_size = len(custom_positions)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate grid size and kernel size
        self.num_cells = HexEngine.solve_length(engine_radius)
        
        # Get correspondence lists (reused, not recomputed)
        correspondence_lists = get_list_of_custom_correspondence_lists(
            engine_radius, custom_positions
        )
                
        # Precompute valid mask
        valid_mask = correspondence_lists != -1
        self.register_buffer(
            'valid_mask',
            torch.from_numpy(valid_mask).float()  # (kernel_size, num_cells)
        )

        # Compute safe correspondence lists
        corr_safe = correspondence_lists.copy()
        corr_safe[~valid_mask] = 0  # Replace -1 with 0
        
        # Register as buffer (not a parameter, moves with model to device)
        self.register_buffer(
            'corr_safe',
            torch.from_numpy(corr_safe).long()  # (kernel_size, num_cells)
        )
        
        # Learnable parameters
        # Weight: (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.kernel_size)
        )
        
        if bias:
            # Bias: (out_channels,)
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using correspondence lists.
        
        Args:
            x: Input tensor (B, C_in * L) or (B, L) if C_in=1
            
        Returns:
            Output tensor (B, C_out * L) or (B, L) if C_out=1
        """
        B = x.shape[0]
        
        # Reshape input to (B, C_in, L)
        if self.in_channels == 1 and x.shape[1] == self.num_cells:
            x = x.unsqueeze(1)  # (B, 1, L)
        else:
            x = x.view(B, self.in_channels, self.num_cells)  # (B, C_in, L)
        
        # Apply convolution using correspondence lists
        out = self._conv_by_correspondence_lists(x)  # (B, C_out, L)
        
        # Reshape output
        if self.out_channels == 1:
            out = out.squeeze(1)  # (B, L)
        else:
            out = out.view(B, -1)  # (B, C_out * L)
        
        return out
    
    def _conv_by_correspondence_lists(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core convolution operation using correspondence lists.
        
        Args:
            x: Input tensor of shape (B, C_in, L)
            
        Returns:
            Output tensor of shape (B, C_out, L)
        """
        B, C_in, L = x.shape
        C_out, _, K = self.weight.shape
        
        # Get correspondence lists: (K, L)
        corr_safe = self.corr_safe
        valid_mask = self.valid_mask
        
        # Gather values for all kernel positions
        # Expand x to (B, C_in, K, L) then gather
        x_expanded = x.unsqueeze(2).expand(-1, -1, K, -1)  # (B, C_in, K, L)
        corr_expanded = corr_safe.unsqueeze(0).unsqueeze(0).expand(B, C_in, -1, -1)  # (B, C_in, K, L)
        
        x_vals = torch.gather(x_expanded, 3, corr_expanded)  # (B, C_in, K, L)
        
        # Zero out invalid positions
        valid_expanded = valid_mask.unsqueeze(0).unsqueeze(0).expand(B, C_in, -1, -1)
        x_vals = x_vals * valid_expanded  # (B, C_in, K, L)
        
        # Apply weights: (C_out, C_in, K) × (B, C_in, K, L) → (B, C_out, L)
        weight_exp = self.weight.unsqueeze(0).unsqueeze(-1)  # (1, C_out, C_in, K, 1)
        x_vals_exp = x_vals.unsqueeze(1)  # (B, 1, C_in, K, L)
        
        # Multiply and sum over C_in and K
        out = (x_vals_exp * weight_exp).sum(dim=2).sum(dim=2)  # (B, C_out, L)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)  # (1, C_out, 1)
        
        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'engine_radius={self.engine_radius}, '
                f'kernel_radius={self.kernel_radius}, '
                f'num_cells={self.num_cells}, '
                f'kernel_size={self.kernel_size}, '
                f'kernel_positions={self.custom_positions}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'bias={self.bias is not None}')
    

class HexShrink(nn.Module):
    """
    Hexagonal shrinking layer that reduces the hexagonal grid radius by n.

    This layer shrinks a hexagonal grid by removing the outermost layers,
    effectively reducing the grid's radius. It has no trainable parameters
    and performs only tensor reshaping and rearrangement based on precomputed
    correspondence lists. The operation maps cells from the original larger
    grid to the smaller grid by shifting coordinates inward.

    Args:
        engine_radius (int): Radius of the original hexagonal grid
        shrink_by (int): Number of layers to remove from the grid boundary.
                         Must be positive and result in a positive final radius.

    Shape:
        - Input: (batch_size, channels, num_cells)
        - Output: (batch_size, channels, new_num_cells)

    Examples:
        >>> # Shrink a radius-5 grid (61 cells) to radius-3 (19 cells)
        >>> shrink = HexShrink(engine_radius=5, shrink_by=2)
        >>> features = torch.rand(8, 16, 61)  # batch=8, channels=16
        >>> shrunk = shrink(features)          # (8, 16, 19)

        >>> # Use in a downsampling network
        >>> model = nn.Sequential(
        ...     PureHexConv(5, 2, 1, 32),
        ...     nn.ReLU(),
        ...     HexShrink(5, 1),  # radius 5 → 4
        ...     PureHexConv(4, 1, 32, 64),
        ...     nn.ReLU(),
        ...     HexShrink(4, 2),  # radius 4 → 2
        ...     PureHexConv(2, 1, 64, 1)
        ... )
    """

    def __init__(self, engine_radius: int, shrink_by: int):
        super(HexShrink, self).__init__()
        if shrink_by <= 0:
            raise ValueError("shrink_by must be positive.")
        if engine_radius - shrink_by <= 0:
            raise ValueError("Resulting engine_radius must be positive.")
        self.engine_radius = engine_radius
        self.shrink_by = shrink_by
        self.new_engine_radius = engine_radius - shrink_by
        self.num_cells = HexEngine.solve_length(engine_radius)
        self.new_num_cells = HexEngine.solve_length(self.new_engine_radius)
        
        # Make correspondence list for shrinking
        correspondence_list = HexShrink._build_shrinking_correspondence_list(
            engine_radius, shrink_by
        )  # (1, num_cells)

        # Precompute valid mask
        valid_mask = correspondence_list != -1
        self.register_buffer(
            'valid_mask',
            torch.from_numpy(valid_mask).float()  # (1, num_cells)
        )

        # Compute safe correspondence list
        corr_safe = correspondence_list.copy()
        corr_safe[~valid_mask] = 0  # Replace -1 with 0

        # Register as buffer (not a parameter, moves with model to device)
        self.register_buffer(
            'corr_safe',
            torch.from_numpy(corr_safe).long()  # (1, num_cells)
        )

    def _build_shrinking_correspondence_list(
        engine_radius: int,
        shrink_by: int
    ) -> np.ndarray:
        """
        Builds a correspondence list for shrinking the hexagonal grid.

        This is not provided by hpyhex-rs due the the different radius.

        Args:
            engine_radius (int): Original radius of the hexagonal grid.
            shrink_by (int): Number of layers to shrink by.
        Returns:
            np.ndarray: 2D array of correspondence list for shrinking.
        """
        original_length = HexEngine.solve_length(engine_radius)
        new_engine_radius = engine_radius - shrink_by
        new_length = HexEngine.solve_length(new_engine_radius)

        correspondence_list = -1 * np.ones((1, original_length), dtype=np.int64)

        for new_idx in range(new_length):
            hex_coord = HexEngine.hpyhex_rs_coordinate_block(
                new_engine_radius, new_idx
            )
            original_hex_coord = hex_coord.shift_i(shrink_by).shift_k(shrink_by)
            original_idx = HexEngine.hpyhex_rs_index_block(
                engine_radius, original_hex_coord
            )
            correspondence_list[0, original_idx] = new_idx

        return correspondence_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to shrink the hexagonal grid.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L_new)
        """
        B, C, L = x.shape
        device = x.device

        if L != self.num_cells:
            raise ValueError(f"Input length L={L} does not match expected num_cells={self.num_cells}.")

        # Prepare output tensor
        out = torch.zeros((B, C, self.new_num_cells), device=device)

        # Prepare an index tensor for advanced indexing
        # Shape: (1, L)
        corr = self.corr_safe.to(device)  # (1, L)

        # Mask for valid indices
        valid_mask = self.valid_mask.to(device) # (1, L)

        # Zero out invalid positions
        x_masked = x * valid_mask  # (B, C, L)

        # For each new cell, sum over all original cells that map to it
        # corr: (1, L), values in [0, new_num_cells-1] or 0 for invalid
        # valid_mask: (1, L), 1 for valid, 0 for invalid

        # Expand corr to (B, C, L) for broadcasting
        corr_expanded = corr.expand(x.shape[0], x.shape[1], -1)  # (B, C, L)

        # Flatten batch and channel dims for easier indexing
        x_flat = x_masked.reshape(-1, L)         # (B*C, L)
        corr_flat = corr_expanded.reshape(-1, L) # (B*C, L)

        # Output: (B*C, new_num_cells)
        out_flat = torch.zeros(x_flat.shape[0], self.new_num_cells, device=device)

        # Use scatter_add to sum values for each new cell index
        out_flat.scatter_add_(1, corr_flat, x_flat)

        # If all invalids are mapped to 0, subtract out the invalids at index 0
        invalid_mask = (valid_mask == 0).expand(x.shape[0], x.shape[1], -1).reshape(-1, L)
        out_flat[:, 0] -= (x_flat * invalid_mask).sum(dim=1)

        # Reshape back to (B, C, new_num_cells)
        out = out_flat.view(B, C, self.new_num_cells)

        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'engine_radius={self.engine_radius}, '
                f'shrink_by={self.shrink_by}, '
                f'new_engine_radius={self.new_engine_radius}, '
                f'num_cells={self.num_cells}, '
                f'new_num_cells={self.new_num_cells}')  


class HexMove(nn.Module):
    """
    Hexagonal moving layer that shifts the hexagonal grid by a given Hex offset.

    This layer moves a hexagonal grid by shifting all cell coordinates
    according to a specified Hex offset. It has no trainable parameters
    and performs only tensor reshaping and rearrangement based on precomputed
    correspondence lists. The operation maps cells from the original grid
    to new positions based on the offset, filling in invalid positions with zeros.

    Moving does not shrink or expand the grid; cells that move outside the
    original grid are discarded, and empty cells are filled with zeros.

    Args:
        engine_radius (int): Radius of the hexagonal grid
        move_by (Hex): Hex offset to move the grid by.

    Shape:
        - Input: (batch_size, channels, num_cells)
        - Output: (batch_size, channels, num_cells)

    Examples:
        >>> # Move a radius-5 grid (61 cells) by Hex(1,-1)
        >>> move = HexMove(engine_radius=5, move_by=Hex(1,-1))
        >>> features = torch.rand(8, 16, 61)  # batch=8, channels=16
        >>> moved = move(features)             # (8, 16, 61)
    """

    def __init__(self, engine_radius: int, move_by: Hex):
        super(HexMove, self).__init__()
        self.engine_radius = engine_radius
        self.move_by = move_by
        self.num_cells = HexEngine.solve_length(engine_radius)

        # Make correspondence list for moving
        correspondence_list = HexEngine.to_numpy_correspondence_list_int64(
            engine_radius, move_by
        )  # (1, num_cells)

        # Precompute valid mask
        valid_mask = correspondence_list != -1
        self.register_buffer(
            'valid_mask',
            torch.from_numpy(valid_mask).float()  # (1, num_cells)
        )

        # Compute safe correspondence list
        corr_safe = correspondence_list.copy()
        corr_safe[~valid_mask] = 0  # Replace -1 with 0

        # Register as buffer (not a parameter, moves with model to device)
        self.register_buffer(
            'corr_safe',
            torch.from_numpy(corr_safe).long()  # (1, num_cells)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to move the hexagonal grid.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L)
        """
        B, C, L = x.shape
        device = x.device

        if L != self.num_cells:
            raise ValueError(f"Input length L={L} does not match expected num_cells={self.num_cells}.")

        # Prepare output tensor
        out = torch.zeros((B, C, self.num_cells), device=device)

        # Prepare an index tensor for advanced indexing
        # Shape: (1, L)
        corr = self.corr_safe.to(device)  # (1, L)

        # Mask for valid indices
        valid_mask = self.valid_mask.to(device) # (1, L)

        # Zero out invalid positions
        x_masked = x * valid_mask  # (B, C, L)

        # For each cell, map to new position
        corr_expanded = corr.expand(x.shape[0], x.shape[1], -1)  # (B, C, L)

        # Flatten batch and channel dims for easier indexing
        x_flat = x_masked.reshape(-1, L)         # (B*C, L)
        corr_flat = corr_expanded.reshape(-1, L) # (B*C, L)

        # Output: (B*C, num_cells)
        out_flat = torch.zeros(x_flat.shape[0], self.num_cells, device=device)

        # Use scatter_add to place values at new positions
        out_flat.scatter_add_(1, corr_flat, x_flat)

        # If all invalids are mapped to 0, subtract out the invalids at index 0
        invalid_mask = (valid_mask == 0).expand(x.shape[0], x.shape[1], -1).reshape(-1, L)
        out_flat[:, 0] -= (x_flat * invalid_mask).sum(dim=1)

        # Reshape back to (B, C, num_cells)
        out = out_flat.view(B, C, self.num_cells)

        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'engine_radius={self.engine_radius}, '
                f'move_by={self.move_by}, '
                f'num_cells={self.num_cells}')


class HexDense(nn.Module):
    """
    Dense (fully connected) layer for hexagonal board states.
    
    Applies a linear transformation to the entire board vector, connecting
    all input cells/channels to all output cells/channels.
    
    Args:
        engine_radius (int): Radius of the hexagonal grid
        in_channels (int): Number of input features per cell
        out_channels (int): Number of output features per cell
        bias (bool): If True, adds a learnable bias per output feature
        
    Shape:
        - Input: (batch_size, in_channels * num_cells) or (batch_size, num_cells) if in_channels=1
        - Output: (batch_size, out_channels * num_cells) or (batch_size, num_cells) if out_channels=1
        
    Examples:
        >>> # Dense layer for board features
        >>> dense = HexDense(engine_radius=5, in_channels=16, out_channels=32)
        >>> features = torch.rand(8, 16*61)  # batch=8, 61 cells
        >>> output = dense(features)         # (8, 32*61)
    """
    
    def __init__(
        self,
        engine_radius: int,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True
    ):
        super(HexDense, self).__init__()

        if engine_radius <= 0:
            raise ValueError("engine_radius must be positive.")
        
        self.engine_radius = engine_radius
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_cells = HexEngine.solve_length(engine_radius)
        
        # Dense layer: (in_channels * num_cells) -> (out_channels * num_cells)
        self.linear = nn.Linear(
            in_channels * self.num_cells,
            out_channels * self.num_cells,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dense layer.
        
        Args:
            x: Input tensor (B, C_in * L) or (B, L) if C_in=1
            
        Returns:
            Output tensor (B, C_out * L) or (B, L) if C_out=1
        """
        B = x.shape[0]
        
        # Ensure input is flattened
        if self.in_channels == 1 and x.shape[1] == self.num_cells:
            x = x.view(B, -1)  # (B, L)
        else:
            x = x.view(B, -1)  # (B, C_in * L)
        
        # Apply linear transformation
        out = self.linear(x)  # (B, C_out * L)
        
        # Reshape output
        if self.out_channels == 1:
            out = out.view(B, self.num_cells)  # (B, L)
        else:
            out = out.view(B, self.out_channels, self.num_cells)  # (B, C_out, L)
        
        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'engine_radius={self.engine_radius}, '
                f'num_cells={self.num_cells}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'bias={self.linear.bias is not None}')


class HexBias(nn.Module):
    """
    Bias layer that adds learnable bias to each cell in the hexagonal grid.
    
    This layer adds a separate bias term to each cell, with options for
    per-channel biases.
    
    Args:
        engine_radius (int): Radius of the hexagonal grid
        channels (int): Number of feature channels per cell
        
    Shape:
        - Input: (batch_size, channels * num_cells) or (batch_size, num_cells) if channels=1
        - Output: Same as input
        
    Examples:
        >>> # Add bias to single-channel board
        >>> bias_layer = HexBias(engine_radius=5, channels=1)
        >>> board = torch.rand(8, 61)
        >>> biased = bias_layer(board)  # (8, 61)
        
        >>> # Add bias to multi-channel features
        >>> bias_layer = HexBias(engine_radius=5, channels=16)
        >>> features = torch.rand(8, 16*61)
        >>> biased = bias_layer(features)  # (8, 16*61)
    """
    
    def __init__(self, engine_radius: int, channels: int = 1):
        super(HexBias, self).__init__()

        if engine_radius <= 0:
            raise ValueError("engine_radius must be positive.")
        if channels <= 0:
            raise ValueError("channels must be positive.")
        
        self.engine_radius = engine_radius
        self.channels = channels
        self.num_cells = HexEngine.solve_length(engine_radius)
        
        # Bias: (channels, num_cells)
        self.bias = nn.Parameter(torch.zeros(channels, self.num_cells))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass adding bias to each cell.
        
        Args:
            x: Input tensor (B, C * L) or (B, L) if C=1
            
        Returns:
            Output tensor with bias added
        """
        B = x.shape[0]
        
        # Reshape to (B, C, L)
        if self.channels == 1 and x.shape[1] == self.num_cells:
            x = x.unsqueeze(1)  # (B, 1, L)
        else:
            x = x.view(B, self.channels, self.num_cells)  # (B, C, L)
        
        # Add bias
        out = x + self.bias.unsqueeze(0)  # (B, C, L)
        
        # Reshape back
        if self.channels == 1:
            out = out.squeeze(1)  # (B, L)
        else:
            out = out.view(B, -1)  # (B, C * L)
        
        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'engine_radius={self.engine_radius}, '
                f'channels={self.channels}, '
                f'num_cells={self.num_cells}')


class HexMerge(nn.Module):
    """
    Bilinear merge layer for hexagonal board states.
    
    Merges two input tensors of the same radius using bilinear weights.
    The weights are universal across all cells in the board.
    
    Args:
        engine_radius (int): Radius of the hexagonal grid
        in_channels_a (int): Number of input channels for first tensor
        in_channels_b (int): Number of input channels for second tensor
        out_channels (int): Number of output channels
        
    Shape:
        - Input x: (batch_size, in_channels_a * num_cells) or (batch_size, num_cells) if in_channels_a=1
        - Input y: (batch_size, in_channels_b * num_cells) or (batch_size, num_cells) if in_channels_b=1
        - Output: (batch_size, out_channels * num_cells) or (batch_size, num_cells) if out_channels=1
        
    Examples:
        >>> # Merge two feature maps
        >>> merge = HexMerge(engine_radius=5, in_channels_a=16, in_channels_b=8, out_channels=32)
        >>> features_a = torch.rand(8, 16*61)
        >>> features_b = torch.rand(8, 8*61)
        >>> output = merge(features_a, features_b)  # (8, 32*61)
    """
    
    def __init__(
        self,
        engine_radius: int,
        in_channels_a: int = 1,
        in_channels_b: int = 1,
        out_channels: int = 1
    ):
        super(HexMerge, self).__init__()

        if engine_radius <= 0:
            raise ValueError("engine_radius must be positive.")
        if in_channels_a <= 0 or in_channels_b <= 0 or out_channels <= 0:
            raise ValueError("All channel counts must be positive.")
        
        self.engine_radius = engine_radius
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.out_channels = out_channels
        self.num_cells = HexEngine.solve_length(engine_radius)
        
        # Bilinear weights: (in_channels_a, in_channels_b, out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels_a, in_channels_b, out_channels)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass performing bilinear merge.
        
        Args:
            x: First input tensor (B, C_a * L) or (B, L) if C_a=1
            y: Second input tensor (B, C_b * L) or (B, L) if C_b=1
            
        Returns:
            Output tensor (B, C_out * L) or (B, L) if C_out=1
        """
        B = x.shape[0]
        
        # Reshape x to (B, C_a, L)
        if self.in_channels_a == 1 and x.shape[1] == self.num_cells:
            x = x.unsqueeze(1)  # (B, 1, L)
        else:
            x = x.view(B, self.in_channels_a, self.num_cells)  # (B, C_a, L)
        
        # Reshape y to (B, C_b, L)
        if self.in_channels_b == 1 and y.shape[1] == self.num_cells:
            y = y.unsqueeze(1)  # (B, 1, L)
        else:
            y = y.view(B, self.in_channels_b, self.num_cells)  # (B, C_b, L)
        
        # Bilinear merge: (B, C_a, L) and (B, C_b, L) with weights (C_a, C_b, C_out) -> (B, C_out, L)
        out = torch.einsum('bal,bbl,abo->bol', x, y, self.weights)
        
        # Reshape output
        if self.out_channels == 1:
            out = out.squeeze(1)  # (B, L)
        else:
            out = out.view(B, -1)  # (B, C_out * L)
        
        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'engine_radius={self.engine_radius}, '
                f'num_cells={self.num_cells}, '
                f'in_channels_a={self.in_channels_a}, '
                f'in_channels_b={self.in_channels_b}, '
                f'out_channels={self.out_channels}')

