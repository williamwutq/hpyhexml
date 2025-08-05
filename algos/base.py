'''
Provide the fundamental algorithms for the HappyHex game.
These algorithms are designed to be used as baseline algorithms for benchmarking and comparison with more complex strategies.
These algorithms include:
- `random`: A random algorithm that selects a piece and position randomly.
- `first`: A first selection algorithm that selects the first valid piece and position.
'''

from hpyhex.hex import HexEngine, Piece, Hex
import random as rd

__all__ = ['random', 'first']

def random(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    '''
    A random algorithm that returns the first piece and position.
    This algorithm randomly selects a piece from the queue and a valid position for that piece.
    If no valid positions are found, it raises a ValueError.

    Parameters:
        engine (HexEngine): The HexEngine instance to use for checking valid positions.
        queue (list[Piece]): The list of pieces available to play.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the selected piece and the position (Hex coordinate) to place it.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    if not queue:
        raise ValueError("Queue is empty")
    else:
        options = []
        seen_pieces = {}
        for piece_index, piece in enumerate(queue):
            key = int(piece)
            if key in seen_pieces: continue
            seen_pieces[key] = piece_index
            for coord in engine.check_positions(piece):
                options.append((piece_index, coord))
        if not options:
            raise ValueError("No valid options found")
        return rd.choice(options)
    
def first(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    '''
    A first selection algorithm that returns the first valid piece and position.
    This algorithm checks the first piece in the queue and returns the first valid position for that piece.
    If no valid positions are found, it raises a ValueError.
    
    Parameters:
        engine (HexEngine): The HexEngine instance to use for checking valid positions.
        queue (list[Piece]): The list of pieces available to play.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the selected piece and the position (Hex coordinate) to place it.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the first piece in the queue.
    '''
    if not queue:
        raise ValueError("Queue is empty")
    result_index = 0
    for piece_index, piece in enumerate(queue):
        positions = engine.check_positions(piece)
        if positions:
            result_index = piece_index
            coord = positions[0]
            break
    else:
        raise ValueError("No valid options found")
    return (result_index, coord)