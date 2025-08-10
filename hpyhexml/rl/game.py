'''
A batched game evironement for HappyHex, designed to be used with reinforcement learning algorithms.

This module provides a way to run multiple games in parallel, allowing for efficient training of reinforcement learning models.
'''

from hpyhex.hex import HexEngine
from hpyhex.game import PieceFactory
from copy import deepcopy as clone

__all__ = ['BatchedGame']


class BatchedGame:
    '''
    A batched game environment for HappyHex for running multiple games from the same initial condition in parallel.

    Comparing to the Game in hpyhex.game, this class does not have properties and does not store absolute turn and score.
    Instead of allowing direct piece placements, only algorithms packaged in specific signature can be used to play the game and receive feedback.
    
    This class is callable, meaning you can and should use it as a function to play the game with a given algorithm and feedback function.
    The signature of the callable is as follows:
    ```python
    def __call__(self, batched_algorithm, batched_feedback, limit=None) -> None:
    ```
    For details on the parameters and expected behavior, please refer to the __doc__ of the __call__ method.
    '''
    def __init__(self, parallel_games: int, queue_length: int, engine: HexEngine, max_batch_size: int = None) -> None:
        '''
        Initialize the BatchedGame with the number of parallel games, queue length, and initial engine.

        During initialization, the game will create a list of HexEngine instances and a list of queues for each game.
        Each game will have its own engine and queue, allowing for independent game states.

        It is recommended to use the default max_batch_size, which is the same as parallel_games, as it allows maximum effective parallelism.
        
        Parameters:
            parallel_games (int): The number of games to run in parallel.
            queue_length (int): The length of the queue for each game.
            engine (HexEngine): The HexEngine instance to use as initial condition for the games.
            max_batch_size (int, optional): The maximum batch size for processing. Defaults to parallel_games.
        '''
        if not isinstance(parallel_games, int) or parallel_games <= 0:
            raise ValueError("parallel_games must be a positive integer.")
        if not isinstance(queue_length, int) or queue_length <= 0:
            raise ValueError("queue_length must be a positive integer.")
        if not isinstance(engine, HexEngine):
            raise ValueError("engine must be an instance of HexEngine.")
        self.parallel_games = parallel_games
        self.max_batch_size = max_batch_size or parallel_games
        self.engines = [clone(engine) for _ in range(parallel_games)] # Clone the engine for each game
        self.queues = [
            [PieceFactory.generate_piece() for _ in range(queue_length)]
            for _ in range(parallel_games)
        ] # Initialize queues with random pieces
        self.moves = [0] * parallel_games  # Track moves made in each game
        # Check if the game is active
        self.active = []
        for i in range(parallel_games):
            for p in self.queues[i]:
                if self.engines[i].check_positions(p):
                    self.active.append(True)
                    break
            else:
                self.active.append(False) # If no piece is found, the game is inactive

    def __call__(self, batched_algorithm, batched_feedback, limit=None) -> None:
        '''
        Play the batched game using the provided algorithm and feedback function.

        The passed in algorithm must be packaged in the following signature:
        ```python
        def batched_algorithm(engines: list[HexEngine], queues: list[list[Piece]]) -> list[tuple[int, Hex]]
        ```
        in which `engines` is a list of engine instances and `queues` is a list of queues for each game.
        The algorithm should return a list of moves, where each move is a tuple of (piece_index, Hex position), representing the index of the piece
        in the queue and the hexagonal coordinate for each game.

        The passed in feedback function must be packaged in the following signature:
        ```python
        def batched_feedback(results: list[tuple[int, int, int]]) -> None:
        ```
        in which `results` is a list of tuples, where each tuple contains (delta_turn, delta_score, total_moves) for each game.

        Although the function implements type checks during initialization, it does not check the types of the algorithm and feedback functions.
        These two functions should implement the abovementioned signatures, otherwise the function may raise unexpected errors during execution.

        Execute this function will run the algorithm on a maximum of `max_batch_size` games at a time, and apply the feedback to the game state.
        It will continue to run until all games are finished or the set limit is reached. If the limit is not set, it will run until all games are finished.
        During each algorithm call, if the returned moves are invalid, the game will end and the delta turn will be set to 0 in the feedback.
        
        Parameters:
            batched_algorithm (callable): A function that takes a list of HexEngine and a list of queues, and returns a list of moves.
            batched_feedback (callable): A function that takes a list of results and applies them to the game state.
            limit (int, optional): The maximum number of moves to play. If None, plays until all games are finished.
        Raises:
            ValueError: If batched_algorithm or batched_feedback is not callable, or if limit is not a positive integer or None.
            TypeError: If anytime in execution, the algorithm returns a move of invalid type.
            Exception: If the algorithm fails unexpectedly.
        '''
        if not callable(batched_algorithm):
            raise ValueError("batched_algorithm must be a callable function.")
        if not callable(batched_feedback):
            raise ValueError("batched_feedback must be a callable function.")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError("limit must be a positive integer or None.")
        batches_done = 0
        while True:
            # Select active games for this step
            active_indices = [i for i, a in enumerate(self.active) if a]
            if not active_indices:
                break  # no games left
            if self.max_batch_size < len(active_indices):
                active_indices = active_indices[:self.max_batch_size]

            # Prepare inputs for algorithm
            active_engines = [self.engines[i] for i in active_indices]
            active_queues = [self.queues[i] for i in active_indices]

            # Ask algorithm for moves
            moves = batched_algorithm(active_engines, active_queues)

            # Validate that moves length matches active_indices length
            if len(moves) != len(active_indices):
                raise ValueError(f"batched_algorithm must return a list of moves with length {len(active_indices)}, but got {len(moves)}.")

            # Apply moves and collect feedback
            delta_results = []
            for idx, (piece_index, coord) in zip(active_indices, moves):
                if not self.active[idx]:
                    # This should never be reached
                    delta_results.append((0, 0, self.moves[idx]))
                    continue

                engine = self.engines[idx]
                queue = self.queues[idx]

                # Try adding piece and run elimination
                try:
                    piece = queue[piece_index]
                    engine.add_piece(coord, piece)
                    delta_turn = 1
                    delta_score = len(piece) + len(engine.eliminate()) * 5
                    queue[piece_index] = PieceFactory.generate_piece()
                except ValueError:
                    # Invalid move ends the game
                    self.active[idx] = False
                    delta_turn = 0
                    delta_score = 0
                else:
                    # Check end condition
                    for p in queue:
                        if engine.check_positions(p):
                            break
                    else:
                        self.active[idx] = False
                self.moves[idx] += delta_turn
                delta_results.append((delta_turn, delta_score, self.moves[idx]))

            # Send feedback
            batched_feedback(delta_results)

            batches_done += 1
            if limit is not None and batches_done >= limit:
                break