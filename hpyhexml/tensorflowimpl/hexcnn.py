'''
A package for implementing hexagonal convolutional neural networks (HexCNN) using TensorFlow and Keras.
This module provides a function to precompute the neighbor index map for a hexagonal grid,
which is essential for defining the convolutional operations in a hexagonal grid structure.
'''

from keras import layers
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

