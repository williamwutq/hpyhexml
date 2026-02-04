try:
    from hpyhex import HexEngine, Piece, Hex
except ImportError:
    raise ImportError(
        "hpyhex-rs is not installed or not accessible." \
        "Please install hpyhex-rs via `pip install hyphex-rs` and ensure hpyhex is uninstalled."
    )

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is not installed. Please install numpy via `pip install numpy`.")

# These cannot fail since they are part of standard library
from random import random, shuffle
from math import exp


__all__ = ['sigmoid_like', 'non_negative', 'non_positive', 'softmax_rank_score',
           'flatten_engine', 'flatten_queue', 'flatten_piece',
           'flatten_single_desired', 'label_single_desired',]


def gaussian(x, c=1.0):
    '''
    Simplified Gaussian function.
    
    Parameters:
        x (float): The input value to evaluate.
        c (float): The inverse of standard deviation of the Gaussian. Default is 1.0.
    Returns:
        float: The output value, which is always between 0.0 and 1.0, and approaches 0 as x increases.
    '''
    return exp(-(x * c) ** 2)


def sigmoid_like(x, k=1.0, p=2.0):
    '''
    Smooth sigmoid-like decay function.
    
    Parameters:
        x (float): The input value to evaluate.
        k (float): The steepness of the curve. Default is 1.0.
        p (float): The power of the decay. Default is 2.0.
    Returns:
        float: The output value, which is between 0.0 and 1.0 and approaches 0 as x increases.
    '''
    return 1.0 / (1.0 + (k * x) ** p)


def non_negative(x):
    '''
    An activation function that activates if x is greater than or equal to 0.
    Returns 1.0 if x is greater than or equal to 0, otherwise returns 0.0.

    Use of this function in a machine learning model is discouraged.

    Use of this function for indexing can guarantee that the output is a non-negative number, but it might be floating point.

    Parameters:
        x (float): The input value to evaluate.
    Returns:
        float: 1.0 if x >= 0, otherwise 0.0.
    '''
    if (x >= 0.0):
        return 1.0
    else:
        return 0.0


def non_positive(x):
    '''
    An activation function that activates if x is less than or equal to 0.
    Returns 1.0 if x is less than or equal to 0, otherwise returns 0.0.

    Use of this function in a machine learning model is discouraged.

    Use of this function for indexing cannot guarantee that the output is one-hot encoded.

    Parameters:
        x (float): The input value to evaluate.
    Returns:
        float: 1.0 if x <= 0, otherwise 0.0.
    '''
    if (x <= 0.0):
        return 1.0
    else:
        return 0.0


def softmax_rank_score(rank, length) -> float:
    '''
    Compute the soft label score for a given rank position in categorical cross-entropy.

    This function is guaranteed to be able to return values that add up to 1.0, if the input and iteration process are correct.

    Scoring Rules:
    - Rank 0 gets the highest score.
    - Each subsequent rank receives half the score of the previous rank (score ∝ 2^-rank-1).
    - The last two ranks are treated equally, sharing the same score.
    - All scores across ranks sum to 1, suitable for soft categorical cross-entropy targets.

    Parameters:
        rank (int): The rank position of the piece, starting from 0.
        length (int): The total number of pieces in the engine.
    Returns:
        score (float): The score for the given rank position.
    Raises:
        ValueError: If length is 0, or if rank is larger than or equal to length.
    '''
    if length == 0:
        raise ValueError("Length cannot be 0")
    elif length == 1:
        return 1.0
    elif rank >= length:
        raise ValueError("Rank cannot be larger than or equal to length")
    elif rank == length - 1:
        return 2 ** -rank
    else:
        return 2 ** (-rank - 1)


def flatten_engine(engine: HexEngine) -> np.ndarray:
    '''
    (**Input**) Flatten a HexEngine instance into a 1D numpy float32 array, where each float represents the occupancy state of engine blocks.s

    Note: This wraps the intrinsic function `HexEngine.to_numpy_float32`.

    Parameters:
        piece (Piece): The HexEngine instance to flatten.
    Returns:
        vector (np.ndarray): A list of floats representing the engine. They are either 0.0 or 1.0.
    Raises:
        TypeError: If the engine is not an instance of HexEngine.
    '''
    if not isinstance(engine, HexEngine):
        raise TypeError("engine must be an instance of HexEngine")
    return engine.to_numpy_float32()


def flatten_queue(queue: list[Piece]) -> np.ndarray:
    """
    (**Input**) Flatten the queue of pieces into a 1D numpy float32 array.

    Note: This wraps the intrinsic function `Piece.vec_to_numpy_float32_flat`.

    Parameters:
        queue (list[Piece]): A list of Piece instances to flatten.
    Returns:
        vector (np.ndarray): A list of floats representing the queue. They are either 0.0 or 1.0.
    Raises:
        TypeError: If the queue is not a list of Piece instances.
    """
    return Piece.vec_to_numpy_float32_flat(queue)


def flatten_piece(piece: Piece) -> np.ndarray:
    '''
    (**Input**) Flatten a single Piece into a 1D numpy float32 array.

    Note: This wraps the intrinsic function `Piece.to_numpy_float32`.

    Parameters:
        piece (Piece): The Piece instance to flatten.
    Returns:
        vector (np.ndarray): A list of floats representing the Piece. They are either 0.0 or 1.0.
    Raises:
        TypeError: If the piece is not an instance of Piece.
    '''
    if not isinstance(piece, Piece):
        raise TypeError("piece must be an instance of Piece")
    return piece.to_numpy_float32()





if __name__ == "__main__":
    test = lambda func, arguments: print(f"Testing {func.__name__}... Result: {func(arguments)}")
    # Test
    print("This is a module containing utility functions for hpyhex-rs and hpyhexml.")
    test(flatten_engine, HexEngine(8))
    test(flatten_queue, [Piece(3), Piece(61)])
    test(flatten_piece, Piece(67))
