'''
A package for implementing hexagonal convolutional neural networks (HexCNN) using TensorFlow and Keras.
This module provides a function to precompute the neighbor index map for a hexagonal grid,
which is essential for defining the convolutional operations in a hexagonal grid structure.
'''

import tensorflow as tf
from keras.layers import Layer
from hpyhex.hex import Hex, HexEngine

def precompute_neighbor_index_map(radius: int) -> dict[int, tuple[int, int, int, int, int, int, int]]:
    '''
    Precompute the neighbor index map for a hexagonal grid of a given radius.

    This function uses a HexEngine instance to calculate the indices of neighboring hexes
    for each hex in the grid. The neighbors are defined as the six hexes surrounding a
    center hex, plus the center hex itself, creating a kernel of size 7.

    Parameters:
        radius (int): The radius of the hexagonal grid.
    Returns:
        map (dict[int, tuple[int, int, int, int, int, int, int]]): A dictionary mapping each hex index to
        a tuple of indices of its neighbors. The tuple contains the indices of the six neighbors
        and the index of the center hex itself. If a neighbor does not exist (i.e., it is outside the grid),
        its index will be -1.
    '''
    engine = HexEngine(radius)
    neighbor_map = {}
    positions = (
        Hex(-1, -1), Hex(-1, 0), Hex(0, -1), Hex(0, 0), Hex(0, 1), Hex(1, 0), Hex(1, 1)
    )
    for index in range(len(engine)):
        center_hex = engine.coordinate_block(index)
        neighbors = []
        for offset in positions:
            neighbor_hex = center_hex + offset
            neighbor_idx = engine.index_block(neighbor_hex)
            neighbors.append(neighbor_idx if neighbor_idx >= 0 else -1)
        neighbor_map[index] = tuple(neighbors)
    return neighbor_map

def get_radius_from_blocks(length: int) -> int:
    '''
    Get the radius of the hexagonal grid based on the number of blocks.

    This function calculates the radius of a hexagonal grid given the number of blocks.
    The radius is determined by solving the equation for the number of blocks in a hexagonal grid.

    Parameters:
        length (int): The number of blocks in the hexagonal grid.
    Returns:
        int: The radius of the hexagonal grid.
    '''
    return HexEngine.solve_radius(length)


class HexConv(Layer):
    '''
    A Keras layer that implements hexagonal convolution using precomputed neighbor indices.
    This layer applies a hexagonal convolution operation to the input tensor, using a fixed kernel size of 7,
    which corresponds to the six neighbors of a hexagon plus the center hexagon itself.
    The layer uses precomputed neighbor indices to efficiently gather the values from the input tensor
    based on the hexagonal grid structure.
    '''
    maps = {} # Map length to precomputed neighbor index map

    def __init__(self, output_dim, **kwargs):
        '''
        Initialize the HexConv layer with the specified output dimension.

        The output dimension should be defined here as the number of output features for the convolution operation.
        The input dimension is determined by the input shape during the build phase.

        Parameters:
            output_dim (int): The number of output features for the convolution operation.
        Raises:
            ValueError: If the output dimension is not a positive integer.
        '''
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.indices_cache = {}  # Cache for indices to avoid recomputation, starts empty
        self.kernel_size = 7  # fixed, due to 7-point Piece.positions

    def build(self, input_shape):
        '''
        Build the layer with the given input shape and cache neighbor indices for efficient computation.

        This method computes and caches the neighbor indices for the given input shape,
        which is essential for efficient hexagonal convolution operations in the layer.

        The method also initializes the kernel weights and bias for the convolution operation.
        The kernel weights are initialized with a glorot uniform initializer, and the bias is initialized to zeros.
        
        The input_shape is expected to be of the form (batch_size, num_blocks, in_features).

        Parameters:
            input_shape (tuple): The shape of the input tensor.
        Raises:
            ValueError: If the number of blocks in the input tensor is not valid.
        '''
        # input_shape: (batch_size, num_blocks, in_features)
        self.in_features = input_shape[-1]
        num_blocks_int = int(input_shape[1])  # Number of hexagonal blocks in the input

        # One kernel weight per neighbor position
        self.kernel_weights = self.add_weight(
            shape=(self.kernel_size, self.in_features, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="hex_kernel_weights"
        )

        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name="hex_bias"
        )

        if num_blocks_int not in self.maps:
            # If neighbor_map is not precomputed for this number of blocks, compute it
            radius = get_radius_from_blocks(num_blocks_int)
            if radius == -1:
                raise ValueError(f"Invalid number of blocks: {num_blocks_int}. Cannot determine radius.")
            else:
                neighbor_map = precompute_neighbor_index_map(radius)
                self.maps[num_blocks_int] = neighbor_map
        else:
            # Use precomputed map
            neighbor_map = self.maps[num_blocks_int]
        indices_list = []
        for k in range(self.kernel_size):
            indices = [
                neighbor_map[i][k] if i in neighbor_map else -1
                for i in range(num_blocks_int)
            ]
            indices_list.append(tf.constant(indices, dtype=tf.int32))
        self.indices_cache[num_blocks_int] = indices_list


    def call(self, inputs):
        '''
        Apply the hexagonal convolution operation to the input tensor.

        Invalid neighbors (with index -1) are masked to zero in the output.

        Parameters:
            inputs (tf.Tensor): Input tensor of shape (batch_size, num_blocks, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, num_blocks, output_dim).
        Raises:
            ValueError: If the number of blocks in the input tensor is not valid.
        '''
        num_blocks_int = int(inputs.shape[1]) if inputs.shape[1] is not None else tf.shape(inputs)[1]

        # Ensure indices are valid for the current number of blocks
        if num_blocks_int not in self.indices_cache:
            # Recompute map
            if num_blocks_int not in self.maps:
                # If neighbor_map is not precomputed for this number of blocks, compute it
                radius = get_radius_from_blocks(num_blocks_int)
                if radius == -1:
                    raise ValueError(f"Invalid number of blocks: {num_blocks_int}. Cannot determine radius.")
                else:
                    neighbor_map = precompute_neighbor_index_map(radius)
                    self.maps[num_blocks_int] = neighbor_map
            else:
                # Use precomputed map
                neighbor_map = self.maps[num_blocks_int]
            # Recompute indices
            indices_list = []
            for k in range(self.kernel_size):
                indices = [
                    neighbor_map[i][k] if i in neighbor_map else -1
                    for i in range(num_blocks_int)
                ]
                indices_list.append(tf.constant(indices, dtype=tf.int32))
            self.indices_cache[num_blocks_int] = indices_list
        else:
            indices_list = self.indices_cache[num_blocks_int]

        gathered = []  # List of tensors of shape (batch_size, num_blocks, in_features)
        for k in range(self.kernel_size):
            indices = indices_list[k]

            # Replace -1 with 0, we’ll zero it out later
            masked_indices = tf.where(indices >= 0, indices, tf.zeros_like(indices))

            # Gather values from inputs
            selected = tf.gather(inputs, masked_indices, axis=1)

            # Mask out invalid (-1) positions with zero
            mask = tf.cast(tf.expand_dims(indices >= 0, -1), tf.float32)
            selected = selected * mask

            gathered.append(selected)  # shape: (batch, num_blocks, in_features)

        # Apply per-kernel linear transform and sum
        outputs = []
        for k in range(self.kernel_size):
            proj = tf.linalg.matmul(gathered[k], self.kernel_weights[k])  # (batch, num_blocks, output_dim)
            outputs.append(proj)

        result = tf.add_n(outputs) + self.bias
        return result  # shape: (batch, num_blocks, output_dim)


# Add common maps for all instances of HexConv
for r in range(2, 13):
    HexConv.maps[get_radius_from_blocks(r)] = precompute_neighbor_index_map(r)