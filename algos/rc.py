'''
A collection of recursive heuristic algorithms for the HappyHex game.
These algorithms are designed to select the best piece and position based on various heuristics,
such as dense index, score gain, and entropy.
'''

from hpyhex.hex import HexEngine, Piece, Hex
from hpyhex.game import PieceFactory as pf

__all__ = ['rcminimax']

def rcminimax(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A recursive heuristic algorithm that selects the best piece and position based on the current entropy, and other indicators of the game state.
    
    Recursion depth is set to 6.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    if not queue:
        raise ValueError("Queue is empty")
    piece_index, coord, _ = __rcminimax(engine, queue, 6)
    return piece_index, coord

def __rcminimax(engine: HexEngine, queue: list[Piece], depth: int = 0) -> tuple[int, Hex, float]:
    '''
    A recursive heuristic algorithm that selects the best piece and position based on the current entropy, and other indicators of the game state.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
        depth (int): The depth of the recursion. If 0, it will use a base case heuristic.
    Returns:
        result (tuple[int, Hex, float]): A tuple containing the index of the best piece, the best position to place it, and the score of that placement.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    if not queue:
        raise ValueError("Queue is empty")
    if depth == 0:
        return __recurse_base(engine, queue)
    options = []
    seen_pieces = {}
    for piece_index, piece in enumerate(queue):
        key = int(piece)
        if key in seen_pieces: continue
        seen_pieces[key] = piece_index
        for coord in engine.check_positions(piece):
            score = __recurse_rate_score(engine, queue, piece_index, coord, depth)
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    return best_placement

def __recurse_rate_score(engine: HexEngine, queue: list[Piece], piece_index: int, coord: Hex, depth: int) -> float:
    '''
    A helper function to recursively calculate the score of a piece placement.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
        piece_index (int): The index of the piece in the queue.
        coord (Hex): The position to place the piece.
        depth (int): The depth of the recursion.
    Returns:
        score (float): The score of the placement.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    # Constant choice count
    choice_count = 12
    # Copy engine and add the piece
    copy_engine = engine.__copy__()
    copy_engine.add_piece(coord, queue[piece_index])
    copy_queue = queue[:]
    copy_queue[piece_index] = pf.generate_piece()
    score = __recurse_base(copy_engine, copy_queue)[2]
    for c in range(choice_count):
        # Copy queue and get a random piece
        copy_queue = queue[:]
        copy_queue[piece_index] = pf.generate_piece()
        # Get the best placement for this new engine and piece
        try:
            score += __rcminimax(copy_engine, copy_queue, depth - 1)[2]
        except ValueError:
            continue
    return score
    
def __recurse_base(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex, float]:
    '''
    A helper function to recursively find the best piece and position based on a heuristic score. This is the same algorithm as `nrsearch`.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        result (tuple[int, int, Hex]): A tuple containing the index and score of the best piece and the best position to place it.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    options = []
    seen_pieces = {}
    for piece_index, piece in enumerate(queue):
        key = int(piece)
        if key in seen_pieces: continue
        seen_pieces[key] = piece_index
        for coord in engine.check_positions(piece):
            score = engine.compute_dense_index(coord, piece) + len(piece)
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            score += len(copy_engine.eliminate()) / engine.radius
            options.append((piece_index, coord, score))
    if not options:
        return None, None, 0.0
    best_placement = max(options, key=lambda item: item[2])
    return best_placement