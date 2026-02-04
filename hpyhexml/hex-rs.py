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