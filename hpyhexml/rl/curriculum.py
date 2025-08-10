'''
A package for curriculum learning for HappyHex in reinforcement learning environments.

Contains helpers for defining game stages and managing curriculum.
For efficiency, this package provides caching for game stages and curriculum definitions.
'''

from hpyhex.hex import HexEngine
from hpyhex.game import Game, random_engine
from random import random, choice
from copy import deepcopy as clone

curricula = {}
# Format: {name -> (radius, func, cache: [HexEngine...])}

def retrieve_all_from_curriculum(name: str) -> list[HexEngine]:
    '''
    Retrieves all game engines from the curriculum by name.

    Parameters:
        name (str): The name of the curriculum.
    Returns:
        list: A list of HexEngine instances representing the game engines.
    Raises:
        ValueError: If the curriculum name does not exist.
    '''
    if name not in curricula:
        raise ValueError(f"Curriculum '{name}' not found.")
    return curricula[name][2]

def retrieve_random_from_curriculum(name: str) -> HexEngine:
    '''
    Retrieves a random game engine from the curriculum by name.

    Parameters:
        name (str): The name of the curriculum.
    Returns:
        HexEngine: A randomly selected or newly generated HexEngine instance from the curriculum.
    Raises:
        ValueError: If the curriculum name does not exist or if the curriculum function is not callable.
    '''
    if name not in curricula:
        raise ValueError(f"Curriculum '{name}' not found.")
    engine_list = curricula[name][2]
    engine_func = curricula[name][1]
    if not engine_list:
        # If the cache is empty, generate a new engine
        if not callable(engine_func):
            raise ValueError(f"Curriculum function for '{name}' is not available and cache is empty.")
        engine = engine_func(radius = curricula[name][0])
        if not isinstance(engine, HexEngine):
            raise ValueError("Curriculum function must return a HexEngine instance.")
        engine_list.append(engine)
        return engine
    else:
        return choice(engine_list)
    
def generate_engine_to_curriculum(name: str) -> HexEngine:
    '''
    Generates a new game engine and adds it to the curriculum cache.

    Parameters:
        name (str): The name of the curriculum.
    Returns:
        HexEngine: The newly generated HexEngine instance.
    Raises:
        ValueError: If the curriculum name does not exist or if the curriculum function is not callable.
    '''
    if name not in curricula:
        raise ValueError(f"Curriculum '{name}' not found.")
    engine_func = curricula[name][1]
    if not callable(engine_func):
        raise ValueError(f"Curriculum function for '{name}' is not available.")
    engine = engine_func(radius = curricula[name][0])
    if not isinstance(engine, HexEngine):
        raise ValueError("Curriculum function must return a HexEngine instance.")
    if engine not in curricula[name][2]:
        curricula[name][2].append(engine)
    return engine

def generate_engines_to_curriculum(name: str, count: int) -> list[HexEngine]:
    '''
    Generates multiple new game engines and adds them to the curriculum cache.

    It will first attempt to call the curriculum function with the count argument to generate the engines,
    and if fails, default to no argument function call.

    Parameters:
        name (str): The name of the curriculum.
        count (int): The number of HexEngine instances to generate.
    Returns:
        list: A list of newly generated HexEngine instances.
    Raises:
        ValueError: If the curriculum name does not exist or if the curriculum function is not callable.
    '''
    if name not in curricula:
        raise ValueError(f"Curriculum '{name}' not found.")
    engine_func = curricula[name][1]
    engine_radius = curricula[name][0]
    if not callable(engine_func):
        raise ValueError(f"Curriculum function for '{name}' is not available.")
    try:
        engines = engine_func(radius = engine_radius, count = count)
        if not isinstance(engines, list) or not all(isinstance(e, HexEngine) for e in engines):
            raise ValueError("Curriculum function must return a list of HexEngine instances.")
        if len(engines) != count:
            raise ValueError(f"Curriculum function must return exactly {count} HexEngine instances, but got {len(engines)}.")
    except Exception as e:
        engines = [engine_func(radius = engine_radius) for _ in range(count)]
    # Add only engines that are not already in the curriculum cache
    existing_engines = set(curricula[name][2])
    for engine in engines:
        if engine not in existing_engines:
            curricula[name][2].append(engine)
            existing_engines.add(engine)
    return engines

def add_engine_to_curriculum(name: str, engine: HexEngine) -> None:
    '''
    Adds a HexEngine instance to the curriculum cache.
    If the engine already exists in the curriculum, it will not be added again.

    Use with caution, as this function allows manual addition of engines to the curriculum without guaranteeing
    that it can be generated by the curriculum function.

    Parameters:
        name (str): The name of the curriculum.
        engine (HexEngine): The HexEngine instance to add.
    Raises:
        ValueError: If the curriculum name does not exist, or if the engine radius does not match the curriculum radius.
        TypeError: If the provided engine is not an instance of HexEngine.
    '''
    if name not in curricula:
        raise ValueError(f"Curriculum '{name}' not found.")
    if not isinstance(engine, HexEngine):
        raise TypeError("Only HexEngine instances can be added to the curriculum.")
    if engine.radius != curricula[name][0]:
        raise ValueError(f"Engine radius {engine.radius} does not match curriculum radius {curricula[name][0]}.")
    if engine not in curricula[name][2]:
        curricula[name][2].append(engine)

def add_engines_to_curriculum(name: str, engines: list[HexEngine]) -> None:
    '''
    Adds multiple HexEngine instances to the curriculum cache.
    If an engine already exists in the curriculum, it will not be added again.

    Use with caution, as this function allows manual addition of engines to the curriculum without guaranteeing
    that they can be generated by the curriculum function.

    Parameters:
        name (str): The name of the curriculum.
        engines (list[HexEngine]): A list of HexEngine instances to add.
    Raises:
        ValueError: If the curriculum name does not exist, or if any engine radius does not match the curriculum radius.
        TypeError: If any provided engine is not an instance of HexEngine.
    '''
    if name not in curricula:
        raise ValueError(f"Curriculum '{name}' not found.")
    for engine in engines:
        if not isinstance(engine, HexEngine):
            raise TypeError("Only HexEngine instances can be added to the curriculum.")
        if engine.radius != curricula[name][0]:
            raise ValueError(f"Engine radius {engine.radius} does not match curriculum radius {curricula[name][0]}.")
        if engine not in curricula[name][2]:
            curricula[name][2].append(engine)

def create_curriculum(name: str, radius: int, func: callable = None) -> None:
    '''
    Creates a new curriculum with the specified name, radius, and function.

    The random generation function follow the exact signature with the named parameters:
    ```python
    def func(radius: int, count: int = None) -> HexEngine | list[HexEngine]:
    ```
    in which `radius` is the radius of the HexEngine instances to generate,
    and `count` is the number of HexEngine instances to generate (default is None).

    The function should return either a single HexEngine instance when count is None or a list of HexEngine instances.

    The function can also be a function that does not support batch generation, following the signature:
    ```python
    def func(radius: int) -> HexEngine:
    ```
    in which `radius` is the radius of the HexEngine instance to generate.
    In this case, the function will be called with the radius parameter only, and it will return a single HexEngine instance.

    Ideally, the function should ensure randomness in the generated HexEngine instances, but it is not required.
    If the function is not provided, the curriculum can still be created and used, but call any generation functions will raise an error.
    For curriculum without a generation function, you can still manually add HexEngine instances.
    
    Parameters:
        name (str): The name of the curriculum.
        radius (int): The radius of the HexEngine instances in this curriculum.
        func (callable, optional): A function that generates HexEngine instances.
    Raises:
        ValueError: If the curriculum name already exists.
        TypeError: If the provided function does not return a HexEngine instance or a list of HexEngine instances.
    '''
    if name in curricula:
        raise ValueError(f"Curriculum '{name}' already exists.")
    if not callable(func):
        raise TypeError("Curriculum function must be callable.")
    curricula[name] = (radius, func, [])

def remove_curriculum(name: str) -> None:
    '''
    Removes a curriculum by name.

    Parameters:
        name (str): The name of the curriculum to remove.
    Raises:
        ValueError: If the curriculum name does not exist.
    '''
    if name not in curricula:
        raise ValueError(f"Curriculum '{name}' not found.")
    del curricula[name]

# Default curricula
def algo_based_startgame(alg: callable, radius: int, queue_length: int, **kwargs) -> None:
    '''
    Create a curriculum for game start stage for engines of a certain radius based on an algorithm.
    The algorithm should follow the signature:
    ```python
    def algorithm(engines: HexEngine, queues: list[Piece]) -> tuple[int, Hex]
    ```

    Parameters:
        alg (callable): The algorithm to use to play the games.
        radius (int): The radius of the HexEngine instances in this curriculum.
        queue_length (int): The length of the queue for each game for the algorithm to use.
    Keyword Args:
        algorithm_name (str): The name of the algorithm to use in the curriculum. Defaults to the name of the algorithm function.
        move_lower_bound (int): The minimum number of moves to play in the game. Defaults to radius.
        move_upper_bound (int): The maximum number of moves to play in the game. Defaults to radius * 2.
        custom_move_func (callable): A custom function to generate random move times. Defaults to random.
    Raises:
        TypeError: If parameters are not of the expected types.
        ValueError: If the curriculum name already exists.
    '''
    # Parse and validate parameters
    if not isinstance(radius, int) or radius <= 0:
        raise TypeError("radius must be a positive integer.")
    if not isinstance(queue_length, int) or queue_length <= 0:
        raise TypeError("queue_length must be a positive integer.")
    if not callable(alg):
        raise TypeError("alg must be a callable function.")
    algorithm_name = kwargs.get('algorithm_name', alg.__name__)
    if not isinstance(algorithm_name, str):
        algorithm_name = str(algorithm_name)
    if not algorithm_name:
        raise ValueError("algorithm_name must be a non-empty string.")
    curriculum_name = f"startgame_{algorithm_name}_{radius}"
    if curriculum_name in curricula:
        raise ValueError(f"Curriculum '{curriculum_name}' already exists.")
    move_lower_bound = kwargs.get('move_lower_bound', radius)
    if not isinstance(move_lower_bound, int) or move_lower_bound < 0:
        raise TypeError("move_lower_bound must be a non-negative integer.")
    move_upper_bound = kwargs.get('move_upper_bound', radius * 2)
    if not isinstance(move_upper_bound, int) or move_upper_bound < move_lower_bound:
        raise TypeError("move_upper_bound must be a non-negative integer greater than or equal to move_lower_bound.")
    custom_random = kwargs.get('custom_move_func', random)
    if not callable(custom_random):
        raise TypeError("custom_move_func must be a callable function.")
    # Define function
    def startgame_func(radius: int, count: int = None) -> HexEngine | list[HexEngine]:
        '''
        Generate one or more HexEngine instances for the start game stage.

        Parameters:
            radius (int): The radius of the HexEngine instance(s).
            count (int, optional): The number of HexEngine instances to generate. If None, generate one instance.
        Returns:
            HexEngine or list[HexEngine]: A new HexEngine instance, or a list of HexEngine instances if count is specified.
        '''
        def single_engine():
            g = Game(radius, queue_length)
            # Get random move times between the lower and upper bounds
            move_times = round(custom_random() * (move_upper_bound - move_lower_bound) + move_lower_bound)
            # Play the game using the algorithm
            for _ in range(int(move_times)):
                if not g.make_move(alg):
                    break
            return g.engine
        if count is None:
            return single_engine()
        else:
            return [single_engine() for _ in range(count)]
    # Create the curriculum
    create_curriculum(curriculum_name, radius, startgame_func)

def algo_based_endgame(alg: callable, radius: int, queue_length: int, **kwargs) -> None:
    '''
    Create a curriculum for game end stage for engines of a certain radius based on an algorithm.
    The algorithm should follow the signature:
    ```python
    def algorithm(engines: HexEngine, queues: list[Piece]) -> tuple[int, Hex]
    ```

    Parameters:
        alg (callable): The algorithm to use to play the games.
        radius (int): The radius of the HexEngine instances in this curriculum.
        queue_length (int): The length of the queue for each game for the algorithm to use.
    Keyword Args:
        algorithm_name (str): The name of the algorithm to use in the curriculum. Defaults to the name of the algorithm function.
        move_lower_bound (int): The minimum number of moves to play in the game. Defaults to radius.
        move_upper_bound (int): The maximum number of moves to play in the game. Defaults to radius * 2.
        attempt_count (int): The number of attempts for the algorithm to play the game. Defaults to 10.
        pass_percentage (float): The percentage of games that must be ended for the game to be considered endgame. Defaults to 0.5.
        custom_move_func (callable): A custom function to generate random move times. Defaults to random.
    Raises:
        TypeError: If parameters are not of the expected types.
        ValueError: If the curriculum name already exists.
    '''
    # Parse and validate parameters
    if not isinstance(radius, int) or radius <= 0:
        raise TypeError("radius must be a positive integer.")
    if not isinstance(queue_length, int) or queue_length <= 0:
        raise TypeError("queue_length must be a positive integer.")
    if not callable(alg):
        raise TypeError("alg must be a callable function.")
    algorithm_name = kwargs.get('algorithm_name', alg.__name__)
    if not isinstance(algorithm_name, str):
        algorithm_name = str(algorithm_name)
    if not algorithm_name:
        raise ValueError("algorithm_name must be a non-empty string.")
    curriculum_name = f"endgame_{algorithm_name}_{radius}"
    if curriculum_name in curricula:
        raise ValueError(f"Curriculum '{curriculum_name}' already exists.")
    move_lower_bound = kwargs.get('move_lower_bound', radius)
    if not isinstance(move_lower_bound, int) or move_lower_bound < 0:
        raise TypeError("move_lower_bound must be a non-negative integer.")
    move_upper_bound = kwargs.get('move_upper_bound', radius * 2)
    if not isinstance(move_upper_bound, int) or move_upper_bound < move_lower_bound:
        raise TypeError("move_upper_bound must be a non-negative integer greater than or equal to move_lower_bound.")
    attempt_count = kwargs.get('attempt_count', 10)
    if not isinstance(attempt_count, int) or attempt_count <= 0:
        raise TypeError("attempt_count must be a positive integer.")
    pass_percentage = kwargs.get('pass_percentage', 0.5)
    if not isinstance(pass_percentage, float) or not (0 <= pass_percentage <= 1):
        raise TypeError("pass_percentage must be a float between 0 and 1.")
    custom_random = kwargs.get('custom_move_func', random)
    if not callable(custom_random):
        raise TypeError("custom_move_func must be a callable function.")
    # Define function
    def endgame_func(radius: int, count: int = None) -> HexEngine | list[HexEngine]:
        '''
        Generate one or more HexEngine instances for the end game stage.

        Parameters:
            radius (int): The radius of the HexEngine instance(s).
            count (int, optional): The number of HexEngine instances to generate. If None, generate one instance.
        Returns:
            HexEngine or list[HexEngine]: A new HexEngine instance, or a list of HexEngine instances if count is specified.
        '''
        if count is None:
            while True:
                # Get a random engine
                re = random_engine(radius)
                ended_games = 0
                for _ in range(attempt_count):
                    g = Game(clone(re), queue_length)
                    # Get random move times between the lower and upper bounds
                    move_times = round(custom_random() * (move_upper_bound - move_lower_bound) + move_lower_bound)
                    # Play the game using the algorithm
                    for _ in range(int(move_times)):
                        if not g.make_move(alg):
                            ended_games += 1
                            break
                if ended_games / attempt_count >= pass_percentage:
                    return re
        else:
            engines = []
            while len(engines) < count:
                # Get a random engine
                re = random_engine(radius)
                ended_games = 0
                for _ in range(attempt_count):
                    g = Game(clone(re), queue_length)
                    # Get random move times between the lower and upper bounds
                    move_times = round(custom_random() * (move_upper_bound - move_lower_bound) + move_lower_bound)
                    # Play the game using the algorithm
                    for _ in range(int(move_times)):
                        if not g.make_move(alg):
                            ended_games += 1
                            break
                if ended_games / attempt_count >= pass_percentage:
                    engines.append(re)
            return engines
    # Create the curriculum
    create_curriculum(curriculum_name, radius, endgame_func)