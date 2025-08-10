'''
A package for curriculum learning for HappyHex in reinforcement learning environments.

Contains helpers for defining game stages and managing curriculum.
For efficiency, this package provides caching for game stages and curriculum definitions.
'''

from hpyhex.hex import HexEngine
from random import choice

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

def add_engine_to_curriculum(name: str, engine: HexEngine):
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

def add_engines_to_curriculum(name: str, engines: list[HexEngine]):
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

def create_curriculum(name: str, radius: int, func: callable = None):
    '''
    Creates a new curriculum with the specified name, radius, and function.

    The random generation function follow the exact signature with the named parameters:
    ```python
    def func(radius: int, count: int = None) -> HexEngine | list[HexEngine]:
    ```
    in which `radius` is the radius of the HexEngine instances to generate,
    and `count` is the number of HexEngine instances to generate (default is None).
    The function should return either a single HexEngine instance when count is None or a list of HexEngine instances.

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