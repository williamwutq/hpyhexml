'''
A collection of non-recursive heuristic algorithms for the HappyHex game.
These algorithms are designed to select the best piece and position based on various heuristics,
such as dense index, score gain, and entropy.
'''

from hpyhex.hex import HexEngine, Piece, Hex
from math import exp

__all__ = ['nrminimax', 'nrminimaxmix', 'nrminimaxeven', 'nrentropy', 'nrsearch', 'nrdenseindex', 'nrscoreindex', 'nrnaivescoreindex', 'nrbineliminate', 'nrsearchworst']

def nrminimax(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position based on the current entropy, and other indicators of the game state.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    options = []
    w_current_entropy = engine.compute_entropy() - 0.21
    seen_pieces = {}
    for piece_index, piece in enumerate(queue):
        key = int(piece)
        if key in seen_pieces:
            continue
        seen_pieces[key] = piece_index
        for coord in engine.check_positions(piece):
            # Inline compute_weighted_index
            score = engine.compute_dense_index(coord, piece) * 4
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            score += len(copy_engine.eliminate()) / copy_engine.radius * 5
            x = copy_engine.compute_entropy() - w_current_entropy
            score += 7 / (1 + exp(-3 * x))
            # Append scored option
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    # Choose best option
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrminimaxmix(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position based on the current entropy, and other indicators of the game state.
    This algorithm considers all indicators with different weights.

    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    options = []
    w_current_entropy = engine.compute_entropy() - 0.21
    seen_pieces = {}
    for piece_index, piece in enumerate(queue):
        key = int(piece)
        if key in seen_pieces: continue
        seen_pieces[key] = piece_index
        for coord in engine.check_positions(piece):
            # Inline compute_weighted_index
            score = engine.compute_dense_index(coord, piece) * 12 + len(piece) * 5
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            score += len(copy_engine.eliminate()) / copy_engine.radius * 8.2
            x = copy_engine.compute_entropy() - w_current_entropy
            score += 11.4 / (1 + exp(-3 * x))
            # Append scored option
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrminimaxeven(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position based on the current entropy, and other indicators of the game state.
    This version gives even weights to all indicators.

    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    options = []
    w_current_entropy = engine.compute_entropy() - 0.21
    seen_pieces = {}
    for piece_index, piece in enumerate(queue):
        key = int(piece)
        if key in seen_pieces: continue
        seen_pieces[key] = piece_index
        for coord in engine.check_positions(piece):
            score = engine.compute_dense_index(coord, piece) + len(piece)
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            score += len(copy_engine.eliminate()) / copy_engine.radius * 2
            x = copy_engine.compute_entropy() - w_current_entropy
            score += 1 / (1 + exp(-3 * x))
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrentropy(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position purely based on the current entropy of the game state.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
    Raises:
        ValueError: If the queue is empty or no valid positions are found for the pieces in the queue.
    '''
    options = []
    w_current_entropy = engine.compute_entropy() - 0.21
    seen_pieces = {}
    for piece_index, piece in enumerate(queue):
        key = int(piece)
        if key in seen_pieces: continue
        seen_pieces[key] = piece_index
        for coord in engine.check_positions(piece):
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            copy_engine.eliminate()
            x = copy_engine.compute_entropy() - w_current_entropy
            score = 1 / (1 + exp(-3 * x))
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrsearch(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position based on the dense index, and score gain of the game state.
    This algorithm computes the dense index and score gain for each piece and position, and returns the one with the highest score.
    
    In the nrminimax package, this is the best algorithm to use.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
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
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrdenseindex(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position based on the dense index.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
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
            score = engine.compute_dense_index(coord, piece)
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrnaivescoreindex(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position based on the naive score index.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
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
            score = len(piece)
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            score += len(copy_engine.eliminate()) * 5
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrscoreindex(engine: HexEngine, queue : list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position based on the score index.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
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
            score = len(piece)
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            score += len(copy_engine.eliminate()) / engine.radius
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrbineliminate(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the best piece and position based on the binary chance of elimination.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best piece and the best position to place it.
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
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            if copy_engine.eliminate():
                score = 1
            else:
                score = 0
            options.append((piece_index, coord, score))
    if not options:
        raise ValueError("No valid options found")
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)

def nrsearchworst(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    '''
    A heuristic algorithm that selects the worst piece and position based on the dense index, and score gain of the game state.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the worst piece and the worst position to place it.
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
        raise ValueError("No valid options found")
    best_placement = min(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    return (best_piece_option, best_position_option)