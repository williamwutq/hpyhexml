'''
A collection of recursive heuristic algorithms for the HappyHex game.
These algorithms are designed to select the best piece and position based on various heuristics,
such as dense index, score gain, and entropy.
'''

from hpyhex.hex import HexEngine, Piece, Hex
from hpyhex.game import PieceFactory as pf

__all__ = ['rcminimax', 'rcsearch']

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


import multiprocessing


def rcsearch(engine: HexEngine, queue: list[Piece], **kwargs) -> tuple[int, Hex]:
    '''
    A recursive heuristic algorithm that selects the best piece and position based on 'nrsearch'.
    It choose the best selections from 'nrsearch' and evaluates them deeper using recursion.
    This function is optimized and it choose the best placements from `nrsearch` and evaluates them deeper using recursion.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
        **kwargs:
            Keyword arguments
            - depth (int): The depth of the recursion needed. Default is 2. If input is negative, default to 0.
            - choices (int): The number of choices to use during simulation. Default is 12. If input is not positive, default to 1.
            - processes (int): The number of parallel processes to run at the same time. Default is 2. If input is not positive, default to 1.
    Returns:
        result (tuple[int, Hex, float]): A tuple containing the index of the best piece, the best position to place it, and the score of that placement.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    if not queue:
        raise ValueError("Queue is empty")
    depth = kwargs.pop("depth", 2)
    if depth < 0: depth = 0
    choices = kwargs.pop("choices", 12)
    if choices < 1: choices = 1
    processes = kwargs.pop("processes", 2)
    if processes < 1: processes = 1
    # Compute thread count from the information
    piece_index, coord, _ = __rcsearch(engine, queue, depth, choices, processes)
    return piece_index, coord

def __rcsearch(engine: HexEngine, queue : list[Piece], depth: int = 0, choices: int = 12, processes: int = 1) -> tuple[int, Hex]:
    '''
    A recursive heuristic algorithm that selects the best piece and position based on 'nrsearch'.
    It choose the best selections from 'nrsearch' and evaluates them deeper using recursion.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
        depth (int): The depth of the recursion. If 0, it will use a base case heuristic.
        choices (int): The number of choices to use during simulation.
        processes (int): The number of parallel processes to run at the same time.
    Returns:
        result (tuple[int, Hex, float]): A tuple containing the index of the best piece, the best position to place it, and the score of that placement.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    if not queue:
        # Check queue is not empty
        raise ValueError("Queue is empty")
    # Compute zero depth options
    options = []
    seen_pieces = {}
    for piece_index, piece in enumerate(queue):
        key = int(piece)
        if key in seen_pieces: continue
        seen_pieces[key] = piece_index
        for coord in engine.check_positions(piece):
            score = __rcsearch_score_func(engine, queue[piece_index], coord)
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    # If depth is 0, return the best option
    if depth == 0:
        return max(options, key=lambda item: item[2])
    # Sort options by score and take the top k depending on engine radius and queue size
    k = (engine.radius - 2) * len(queue)
    if len(options) < k * 2:
        k = len(options)
    elif k < 6:
        k = 6
    options.sort(key=lambda item: item[2], reverse=True)
    options = options[:k]
    # Recurse on the top k options
    best_option = None
    best_score = 0.0
    for piece_index, coord, _ in options:
        score = __rcsearch_recurse_rate_score(engine, queue, piece_index, coord, depth, choices, processes)
        if score > best_score:
            best_score = score
            best_option = (piece_index, coord, score)
    return best_option

def __rcsearch_recurse_rate_score(engine: HexEngine, queue: list[Piece], piece_index: int, coord: Hex, depth: int, choices: int, processes: int) -> float:
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
    # Copy engine and add the piece
    copy_engine = engine.__copy__()
    copy_engine.add_piece(coord, queue[piece_index])
    # Construct list
    input_list = [(copy_engine, queue, piece_index, depth) for _ in range(choices)]
    with multiprocessing.Pool(processes=processes) as pool:
        score_map = pool.map(__rcsearch_recurse_rate_score_get, input_list)
        score = sum(score_map)
    return score

def __rcsearch_recurse_rate_score_get(structured_tuple: tuple) -> float:
    '''
    A helper function to calculate the score of a piece placement based on the best score obtained by the best placement/

    Parameters:
        structured_tuple (tuple):
            A tuple with the following structure
            - engine (HexEngine): The game engine.
            - piece (Piece): The piece to be placed.
            - piece_index (int): The index of the piece in the queue.
            - depth (Hex): The depth of the search.
    '''
    engine, queue, piece_index, depth = structured_tuple
    # Copy queue and get a random piece
    copy_queue = queue[:]
    copy_queue[piece_index] = pf.generate_piece()
    try:
        return __rcsearch(engine, copy_queue, depth - 1)[2]
    except ValueError:
        return 0.0

def __rcsearch_score_func(engine: HexEngine, piece: Piece, coord: Hex) -> float:
    '''
    A helper function to calculate the score of a piece placement based on dense index and elimination.
    The logic employed is exactly the same as `nrsearch`'s scoring function.

    The score returned is a float guaranteed to be between 0 and positive infinity.
    It is not recommended to normalize this score.

    Parameters:
        engine (HexEngine): The game engine.
        piece (Piece): The piece to be placed.
        piece_index (int): The index of the piece in the queue.
        coord (Hex): The position to place the piece.
    Returns:
        score (float): The score of the placement.
    '''
    score = engine.compute_dense_index(coord, piece) + len(piece)
    copy_engine = engine.__copy__()
    copy_engine.add_piece(coord, piece)
    score += len(copy_engine.eliminate()) / engine.radius
    return score