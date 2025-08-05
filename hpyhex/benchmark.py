'''
This module provides functions to benchmark and compare different algorithms for the game HappyHex.
It includes functions to benchmark a single algorithm, compare two algorithms, and compare an algorithm with a dataset of game states.
These methods can be used to evaluate the performance of different strategies in the game.
All methods include robust error handling and type checking to ensure correct usage.

Functions:
- benchmark: Benchmark a single algorithm by running multiple games and printing the results.
- compare: Compare two algorithms by running multiple games and printing the results.
- compare_with_data: Compare an algorithm with a dataset of game states and print the accuracy and similarity percentage.
'''

from . import game, hex
from time import time


def benchmark(alg, engine_radius: int, queue_size: int,
              eval_times: int = 10, print_on_completion: bool = True,
              print_on_move: bool = False, print_on_game: bool = False) -> tuple[float, float]:
    '''
    Benchmark the given algorithm by running multiple games and printing the results.

    Parameters:
        alg (callable): The algorithm to benchmark.
        engine_radius (int): The radius of the HexEngine.
        queue_size (int): The size of the queue.
        eval_times (int): The number of turns to run for benchmarking.
        print_on_completion (bool): Whether to print the completion message.
        print_on_move (bool): Whether to print the game state after each move.
        print_on_game (bool): Whether to print the game result after each game.
    Returns:
        result (tuple): The average result of the games.
    Raises:
        TypeError: If any of the parameters are of incorrect type.
    '''
    if not isinstance(eval_times, int) or eval_times < 1:
        raise TypeError("eval_times must be positive integer")
    if not isinstance(engine_radius, int) or engine_radius < 2:
        raise TypeError("engine_radius must be positive integer greater than 1")
    if not isinstance(queue_size, int) or queue_size < 1:
        raise TypeError("queue_size must be positive integer")
    if not isinstance(print_on_completion, bool) or not isinstance(print_on_move, bool) or not isinstance(print_on_game, bool):
        raise TypeError("print settings must be boolean values")
    max_result = (0,0)
    min_result = (1000000000,10000000000)
    sum_result = [0, 0]
    time_start = time()
    for t in range(eval_times):
        g = game.Game(engine_radius, queue_size)
        while not g.end:
            if not g.make_move(alg):
                break
            elif print_on_move:
                print(f"Game {t+1} turn {g.turn} complete")
        result = g.result
        if print_on_game:
            print(f"Game {t+1} result: {g.turn}, {g.score}")
        if result > max_result:
            max_result = result
        if result < min_result:
            min_result = result
        sum_result[0] += result[0]
        sum_result[1] += result[1]
    if print_on_completion:
        print(f"-- Benchmarking complete --")
        print(f"Algorithm used: {alg.__name__}")
        print(f"Total run time: {time() - time_start:.4f} seconds")
        if sum_result[0] == 0:
            avg_time = 0
        else:
            avg_time = (time() - time_start) / sum_result[0]
        print(f"Average   time: {avg_time:.4f} seconds")
        print(f"Max     result: {max_result[0]}, {max_result[1]}")
        print(f"Min     result: {min_result[0]}, {min_result[1]}")
        print(f"Average result: {sum_result[0] / eval_times}, {sum_result[1] / eval_times}")
    return sum_result[0] / eval_times, sum_result[1] / eval_times


def compare(alg1, alg2, engine_radius: int, queue_size: int,
            eval_times: int = 100, print_on_completion = True,
            print_on_move: bool = False, print_on_game: bool = False) -> float:
    '''
    Compare two algorithms by running multiple games and printing the results.

    Parameters:
        alg1 (callable): The first algorithm to compare.
        alg2 (callable): The second algorithm to compare.
        engine_radius (int): The radius of the HexEngine.
        queue_size (int): The size of the queue.
        eval_times (int): The number of turns to run for benchmarking.
        print_on_move (bool): Whether to print the game state after each move.
        print_on_game (bool): Whether to print the game result after each game.
    Returns:
        float: The similarity percentage of the two algorithms based on the number of matches.
    '''
    if not isinstance(eval_times, int) or eval_times < 1:
        raise TypeError("eval_times must be positive integer")
    if not isinstance(engine_radius, int) or engine_radius < 2:
        raise TypeError("engine_radius must be positive intege greater than 1")
    if not isinstance(queue_size, int) or queue_size < 1:
        raise TypeError("queue_size must be positive integer")
    if not isinstance(print_on_completion, bool) or not isinstance(print_on_move, bool) or not isinstance(print_on_game, bool):
        raise TypeError("print settings must be boolean values")
    time_start = time()
    match = 0
    count = 0
    t = 0
    while count < eval_times:
        g = game.Game(engine_radius, queue_size, alg1 if t % 2 == 0 else alg2)
        while not g.end:
            try:
                index1, coord1 = alg1(g.engine, g.queue)
            except ValueError:
                break
            try:
                index2, coord2 = alg2(g.engine, g.queue)
            except ValueError:
                break
            if not g.add_piece(index1 if t % 2 == 0 else index2, coord1 if t % 2 == 0 else coord2):
                break
            if count >= eval_times:
                break
            count += 1
            if index1 == index2 and coord1 == coord2:
                match += 1
            if print_on_move:
                print(f"Game {t+1} turn {g.turn} complete")
        t += 1
        if print_on_game:
            print(f"Game {t} result: {g.turn}, {g.score}")
    if print_on_completion:
        print(f"-- Comparison complete --")
        print(f"Algorithm  one: {alg1.__name__}")
        print(f"Algorithm  two: {alg2.__name__}")
        print(f"Total run time: {time() - time_start:.4f} seconds")
        print(f"Total  Matches: {match}/{eval_times}")
        print(f"Similarity    : {match/eval_times*100:.2f}%")
    return match / eval_times


def compare_with_data(alg, data: list[tuple[hex.HexEngine, list[hex.Piece], list[tuple[int, hex.Hex]]]],
                      data_name: str = "data", rank_limit = None,
                      print_on_completion = True, print_on_move: bool = False) -> float:
    '''
    Compare the algorithm with a dataset of game states.

    Parameters:
        alg (callable): The algorithm to compare.
        data (list): The dataset of game states.
        data_name (str): The name of the dataset.
        rank_limit (int): The maximum number of results to consider for each game state.
        print_on_completion (bool): Whether to print the completion message.
        print_on_move (bool): Whether to print the game state after each move.
    Returns:
        float: The accuracy percentage of the algorithm with the dataset. To obtain the similarity percentage, set rank_limit to 1.
    Raises:
        TypeError: If any of the parameters are of incorrect type.
    '''
    if not isinstance(data, list) or not all(isinstance(item, tuple) and len(item) == 3 for item in data) or not data:
        raise TypeError("data must be a list of tuples (HexEngine, list[Piece], list[tuple[int, Hex]])")
    if not isinstance(data_name, str):
        data_name = str(data_name)
    if rank_limit is not None and (not isinstance(rank_limit, int) or rank_limit < 1):
        raise TypeError("rank_limit must be an positive integer")
    if not isinstance(print_on_completion, bool) or not isinstance(print_on_move, bool):
        raise TypeError("print settings must be boolean values")
    time_start = time()
    count = len(data)
    strict_match = 0
    match = 0
    for i, (engine, queue, results) in enumerate(data):
        try:
            index, coord = alg(engine, queue)
        except Exception:
            continue
        if print_on_move:
            print(f"Turn {i} complete")
        if (index, coord) in results[:rank_limit]:
            match += 1
            if (index, coord) == results[0]:
                strict_match += 1
    if print_on_completion:
        print(f"-- Comparision complete --")
        print(f"Algorithm used: {alg.__name__}")
        print(f"Dataset   name: {data_name}")
        print(f"Total run time: {time() - time_start:.4f} seconds")
        print(f"Total  Matches: {match}/{count}")
        print(f"Strict Matches: {strict_match}/{count}")
        print(f"Accuracy      : {match/count*100:.2f}%")
        print(f"Similarity    : {strict_match/count*100:.2f}%")
    return match / count
