# hpyhex
Simplified implementations of the HappyHex game components and hexagonal system.

## Installation
```bash
pip install hpyhex
```

## Features
- Hexagonal grid representation
- Basic game mechanics for HappyHex
- Utility functions for hexagonal calculations

## Author
Developed by William Wu.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Quickstart

1. Install the package:
   ```bash
   pip install hpyhex
   ```
2. Import and use the main classes as shown above.
3. Create custom algorithms to interact with the game environment.

## Other Implementations

- [hpyhex-rs](https://github.com/williamwutq/hpyhex-rs): A Rust implementation of the `hpyhex` API defined in this package. This implementation claims to offer significant performance improvements over the Python version, but only include the stable core and game API without benchmarking functions or extended API, nor does it plan to include them in the future. See its documentation for more details.

## Usage
```python
from hpyhex.hex import Hex, Piece, HexEngine
from hpyhex.game import Game, PieceFactory

# Create a hexagonal coordinate
coo = Hex(0, 1)

# Create a piece by name
piece = PieceFactory.get_piece("triangle_3_a")

# Create a game engine with radius 3
engine = HexEngine(radius=3)

# Add a piece to the engine
engine.add_piece(piece, coo)

# Eliminate lines and get score
score = len(engine.eliminate()) * 5

# Create a game with engine radius and queue size
game = Game(engine=3, queue=5)
print(game)

# Make a move using a custom algorithm
def simple_algorithm(engine, queue):
	# Always place the first piece at the center
	return 0, Hex(0, 0)
game.make_move(simple_algorithm)
```

## Main Classes

- **Hex**: Represents a hexagonal grid coordinate using a custom line-based system. Supports arithmetic, hashing, and tuple compatibility.
- **Piece**: Represents a shape made of 7 blocks, optimized for memory and performance. Use `PieceFactory` to create pieces by name or byte value.
- **HexEngine**: Manages the hexagonal grid, supports adding pieces, eliminating lines, and computing entropy.
- **PieceFactory**: Utility for creating pieces by name, byte, or randomly. Provides access to all predefined pieces.
- **Game**: Manages the game state, piece queue, score, and turn. Supports adding pieces and making moves with algorithms.

## Hexagonal System

The `Hex` class represents a 2D coordinate in a hexagonal grid system using a specialized integer coordinate model. It supports both raw coordinate access and derived line-based computations across three axes: I, J, and K.

### Coordinate System

This system uses three axes (I, J, K) that run diagonally through the hexagonal grid:

- I+ is 60 degrees from J+, J+ is 60 degrees from K+, and K+ is 60 degrees from I-.
- Coordinates (i, k) correspond to a basis for representing any hexagon.
- **Raw coordinates** (or hex coordinates) refer to the distance of a point along one of the axes multiplied by 2.
- For raw coordinates, the relationships between the axes are defined such that `i - j + k = 0`.
- **Line coordinates** (or line-distance based coordinates) are based on the distance perpendicular to the axes.
- For line coordinates, the relationships between the axes are defined such that `I + J - K = 0`.
- All line coordinates correspond to some raw coordinate, but the inverse is not true. Due to the complexities with dealing with raw coordinates, it is preferable to use line coordinates. **The `hpyhex` API discourages the use of raw coordinates, and all its methods refers to line coordinates only, except those for backward compatibility.**

#### Coordinate System Visualization

Three example points with raw coordinates (2i, 2j, 2k):

```
   I
  / * (5, 4, -1)
 /     * (5, 7, 2)
o - - J
 \ * (0, 3, 3)
  \
   K
```

Three example points with line coordinates (I, J, K):

```
   I
  / * (1, 2, 3)
 /     * (3, 1, 4)
o - - J
 \ * (2, -1, 1)
  \
   K
```

### Grid Structure

- Uses an axial coordinate system (I, K) to represent hexagonal grids, where J = K - I.
- Three axes: I, J, K (not to be confused with 3D coordinates).
- Line-coordinates (I, K) are perpendicular distances to axes, calculated from raw coordinates.

### Grid Size

The total number of blocks in a hexagonal grid of radius `r` is calculated as:

```
Aₖ = 1 + 3*r*(r - 1)
```

This is derived from the recursive pattern:

```
Aₖ = Aₖ₋₁ + 6*(k - 1); A₁ = 1
```

Valid hexagonal grid sizes for common radii:
- Radius 0: 0 cell (Not valid for HexEngine, but can be valid for other purposes)
- Radius 1: 1 cell
- Radius 2: 7 cells
- Radius 3: 19 cells
- Radius 4: 37 cells
- Radius 5: 61 cells
- Radius 6: 91 cells
- Radius 10: 271 cells

### Hex Class Details

Represents a hexagonal grid coordinate using a custom line-based coordinate system.

This class models hexagonal positions with two line coordinates (i, k), implicitly defining the third axis (j) as `j = k - i` to maintain hex grid constraints. It supports standard arithmetic, equality, and hashing operations, as well as compatibility with coordinate tuples.

For small grids, Hex instances are cached for performance, allowing more efficient memory usage and faster access. The caching is limited to a range of -64 to 64 for both i and k coordinates.

Use of Hex over tuples is recommended for clarity and to leverage the singleton feature of small Hexes.

#### Attributes
- `i` (int): The line i coordinate.
- `j` (int): The computed line j coordinate (k - i).
- `k` (int): The line k coordinate.

#### Notes
- This class is immutable and optimized with `__slots__`.
- Raw coordinate methods (`__i__`, `__j__`, `__k__`) are retained for backward compatibility.
- Only basic functionality is implemented; complex adjacency, iteration, and mutability features are omitted for simplicity.



## Benchmarking Algorithms

The package includes benchmarking tools to evaluate and compare algorithms:

```python
from hpyhex.benchmark import benchmark, compare

# Benchmark a single algorithm
avg_score, avg_turn = benchmark(simple_algorithm, engine_radius=3, queue_size=5, eval_times=10)

# Compare two algorithms
similarity = compare(alg1, alg2, engine_radius=3, queue_size=5, eval_times=100)
```

## Detailed API Reference

The following sections describe the `hpyhex` API of version 0.2 in detail.

The **core API** consists of the `Hex`, `Piece`, and `HexEngine` classes, which provide the fundamental building blocks for hexagonal grid operations and game mechanics. The **game API** extends this functionality with `PieceFactory` and `Game`, enabling higher-level game logic and piece management. The `random_engine` function is generally considered part of the game API, though it is often included in core API implementations as well. Benchmarking functions are part of the **extended API** and may be subject to change.

The `hpyhex` API is designed to be Pythonic and intuitive, serving as both a practical toolkit for game operations and a clear reference for similar APIs or implementations in other languages. Its concise, focused design exposes only the essential game mechanics, making it easy to understand, maintain, and extend for core gameplay logic. For example, compared to the original Java implementation—which was long, redundant, and heavily object-oriented—this API omits unnecessary complexity such as a separate `Block` class with color attributes, extra data in `HexEngine` or `Piece`, and special classes for arrays of pieces. By avoiding superfluous extensions intended for GUI integration or enhanced user experience, the API remains streamlined and accessible.

While performance is a core focus, it is balanced with clarity and usability rather than being the sole design priority. The API employs several optimizations: the flyweight pattern in the `Hex` class for efficient coordinate reuse, static initialization and caching in the `Piece` class for memory efficiency, and the use of boolean arrays (rather than arrays of `Hex` objects) in `HexEngine` for fast state management. These choices ensure the API is both performant and robust, providing a solid foundation for extensible and efficient game logic.

### Hex Class

- `__new__(cls, i: Union[int, tuple]=0, k: int=0) -> 'Hex'`: Create a new Hex coordinate at (i, k) or from a tuple. Defaults to (0, 0).
- `__init__(self, i: int=0, k: int=0) -> None`: Initialize a Hex coordinate at (i, k). Defaults to (0, 0).
- `i` (property): Get the I-line coordinate of the hex.
- `j` (property): Get the J-line coordinate of the hex.
- `k` (property): Get the K-line coordinate of the hex.
- `__iter__(self) -> iter`: Return an iterator over the hex coordinates. In python code, use the `for i in hex` syntax.
- `__i__(self) -> int`: Return the raw i coordinate of the hex. Usually not recommended for use.
- `__j__(self) -> int`: Return the raw j coordinate of the hex. Usually not recommended for use.
- `__k__(self) -> int`: Return the raw k coordinate of the hex. Usually not recommended for use.
- `__str__(self) -> str`: Return a string representation of the hex coordinates. In python code, call `str()` instead.
- `__repr__(self) -> str`: Return a string representation of the hex coordinates for debugging. In python code, call `repr()` instead.
- `__eq__(self, value: Union['Hex', tuple]) -> bool`: Check equality with another Hex or a tuple of coordinates.
- `__hash__(self) -> int`: Return a hash of the hex coordinates.
- `__add__(self, other: Union['Hex', tuple]) -> 'Hex'`: Add another Hex or a tuple of coordinates to this hex.
- `__radd__(self, other: Union['Hex', tuple]) -> 'Hex'`: Reverse addition of this hex to another Hex or a tuple.
- `__sub__(self, other: Union['Hex', tuple]) -> 'Hex'`: Subtract another Hex or a tuple of coordinates from this hex.
- `__rsub__(self, other: Union['Hex', tuple]) -> 'Hex'`: Reverse subtraction of this hex from another Hex or a tuple.
- `__copy__(self) -> 'Hex'`: Create a copy of this Hex.
- `__deepcopy__(self, memo=None) -> 'Hex'`: Create a deep copy of this Hex.
- `__bool__(self) -> bool`: Check if the Hex is not at the origin (0, 0). This is compatible with if statements.
- `shift_i(self, units: int) -> 'Hex'`: Return a new Hex shifted along the i-axis by units.
- `shift_j(self, units: int) -> 'Hex'`: Return a new Hex shifted along the j-axis by units.
- `shift_k(self, units: int) -> 'Hex'`: Return a new Hex shifted along the k-axis by units.

### Piece Class

- `__new__(cls, states: Union[list[bool], int] = None) -> 'Piece'`: Create or retrieve a cached Piece instance based on the occupancy states.
- `__init__(self, states: Union[list[bool], int] = None) -> None`: Initialize a Piece with a list of occupancy states.
- `__repr__(self) -> str`: Return a string representation of the Piece in byte format. In python code, call `repr()` instead.
- `__str__(self) -> str`: Return a string representation of the Piece. In python code, call `str()` instead.
- `__iter__(self) -> iter`: Return an iterator over the occupancy states of the Piece. In python code, use the `for state in piece` syntax.
- `__len__(self) -> int`: Return the number of occupied blocks in the Piece. In python code, use the `len()` function.
- `__bool__(self) -> bool`: Check if the Piece has any occupied blocks. This is compatible with if statements.
- `__int__(self) -> int`: Return a byte representation of the blocks in a standard 7-Block piece.
- `states` (property): Get the tuple of boolean values representing the occupancy state of each block in the Piece.
- `coordinates` (property): Get the list of Hex coordinates representing the positions of the blocks in the Piece.
- `__eq__(self, other: Union['Piece', int]) -> bool`: Check if the occupancy states match.
- `__hash__(self) -> int`: Return a hash of the Piece's occupancy states.
- `count_neighbors(self, coo: Union[Hex, tuple]) -> int`: Count occupied neighboring Blocks around the given Hex position.
- `all_pieces(cls) -> list['Piece']`: Get a list of all possible non-empty Piece instances.
- `contigous_pieces(cls) -> list['Piece']`: Get a list of all possible contigous Piece instances.

### HexEngine Class

- `__in_range(coo: Union[Hex, tuple], radius: int) -> bool`: Check if a Hex coordinate is within the specified radius of the hexagonal grid. Although intended for internal use, calling _HexEngine__in_range is possible but not recommended, but if creating a HexEngine is expensive, this method can be used to check coordinates without instantiating a HexEngine.
- `solve_length(radius: int) -> int`: Solves for the length of a HexEngine based on its radius.
- `solve_radius(length: int) -> int`: Solves for the radius of a HexEngine based on its length.
- `__init__(self, arg: Union[int, list[bool], str]) -> None`: Construct a HexEngine with the specified radius.
- `radius` (property): Get the radius of the hexagonal grid.
- `states` (property): Get the occupancy states of the hexagonal grid blocks.
- `__eq__(self, value)`: Check equality with another HexEngine or a list of booleans.
- `__hash__(self) -> int`: Return a hash of the HexEngine's occupancy states.
- `__len__(self) -> int`: Get the number of blocks in the hexagonal grid. In python code, use the `len()` function.
- `__iter__(self) -> iter`: Return an iterator over the occupancy states of the hexagonal grid blocks. In python code, use the `for state in engine` syntax.
- `__repr__(self)`: Return a string representation of the grid block states. In python code, call `repr()` instead.
- `__str__(self) -> str`: Return a string representation of the grid block states. In python code, call `str()` instead.
- `__copy__(self) -> 'HexEngine'`: Create a deep copy of the HexEngine.
- `__deepcopy__(self, memo=None) -> 'HexEngine'`: Create a deep copy of the HexEngine.
- `reset(self) -> None`: Reset the HexEngine grid to its initial state, clearing all blocks.
- `in_range(self, coo: Union[Hex, tuple]) -> bool`: Check if a Hex coordinate is within the radius of the hexagonal grid.
- `index_block(self, coo: Union[Hex, tuple]) -> int`: Get the index of the Block at the specified Hex coordinate.
- `coordinate_block(self, index: int) -> Hex`: Get the Hex coordinate of the Block at the specified index.
- `get_state(self, coo: Union[Hex, tuple, int]) -> bool`: Get the Block occupancy state at the specified Hex coordinate or index.
- `set_state(self, coo: Union[Hex, tuple, int], state: bool) -> None`: Set the occupancy state of the Block at the specified Hex coordinate.
- `check_add(self, coo: Union[Hex, tuple], piece: Union[Piece, int]) -> bool`: Check if a Piece can be added to the hexagonal grid without overlaps.
- `add_piece(self, coo: Union[Hex, tuple], piece: Union[Piece, int]) -> None`: Add a Piece to the hexagonal grid at the specified Hex coordinate.
- `check_positions(self, piece: Union[Piece, int]) -> list[Hex]`: Return all valid positions where another grid can be added.
- `eliminate(self) -> list[Hex]`: Eliminate fully occupied lines along I, J, or K axes and return eliminated coordinates.
- `__eliminate_i(self, eliminate : list[Hex]) -> None`: Identify coordinates along I axis that can be eliminated and insert them into the input list. Do not call this method directly; use `eliminate()` instead.
- `__eliminate_j(self, eliminate : list[Hex]) -> None`: Identify coordinates along J axis that can be eliminated and insert them into the input list. Do not call this method directly; use `eliminate()` instead.
- `__eliminate_k(self, eliminate : list[Hex]) -> None`: Identify coordinates along K axis that can be eliminated and insert them into the input list. Do not call this method directly; use `eliminate()` instead.
- `count_neighbors(self, coo: Union[Hex, tuple]) -> int`: Count occupied neighboring Blocks around the given Hex position.
- `get_pattern(self, coo: Union[Hex, tuple]) -> int`: Determine the pattern of blocks around the given position in the hexagonal grid, including the block itself.
- `compute_dense_index(self, coo: Hex, piece: Piece) -> float`: Compute a density index score for hypothetically placing another piece.
- `compute_entropy(self) -> float`: Compute the entropy of the hexagonal grid based on the distribution of 7-block patterns.
- `all_engines(cls, radius: int) -> list['HexEngine']`: Generate all possible HexEngine instances representing valid occupancy states for a given radius.

### Random Engine Generation

- `random_engine(radius: int) -> HexEngine`: Generate a random HexEngine with a given radius. The engine is in its eliminated form, meaning no full lines exist. The distribution of engines is not guaranteed to be uniform due to elimination, but is near-uniform.

### PieceFactory Class

- `get_piece(name: str) -> Piece`: Get a piece by its name. See [Piece Names Mapping](#piece-names-mapping) for valid names.
- `get_piece_name(p : Piece) -> str`: Get the name of a piece based on its byte value. See [Piece Names Mapping](#piece-names-mapping) for valid names.
- `generate_piece() -> Piece`: Generate a random piece based on frequency distribution. The generation is also used for other standard game implementations across different platforms and program languages.
- `all_pieces() -> tuple[Piece]`: Return all pieces that are defined in this factory. This is different from `Piece.all_pieces()` as this only returns the pieces defined in this factory. This is a subset of all possible pieces and this set is considered `valid` pieces in some contexts, but all pieces returned by `Piece.all_pieces()` are considered valid in general.

### Game Class

- `__init__(self, engine: int, queue: int, initial_turn: int = 0, initial_score: int = 0) -> None`: Initialize the game with a game engine of radius r and game queue of length q.
- `add_piece(self, piece_index: int, coord: Hex) -> bool`: Add a piece to the game engine at the specified coordinates.
- `make_move(self, algorithm) -> bool`: Make a move using the specified algorithm.
- `__str__(self)`: Return a string representation of the game state. In python code, call `str()` instead.
- `__repr__(self) -> str`: Return a string representation of the game state. In python code, call `repr()` instead.
- `end` (property): Returns whether this game has ended. This is useful in a loop:
   ```python
   while not game.end:
         game.make_move(algorithm)
   ```
- `result` (property): Returns the current result of this game.
- `turn` (property): Returns the current turn number of this game.
- `score` (property): Returns the current score of this game.
- `engine` (property): Returns the reference to game engine of this game. This reference is mutable but modifying it may lead to unexpected behavior.
- `queue` (property): Returns the reference to the queue of pieces available in this game. This reference is mutable but modifying it may lead to unexpected behavior. It is **VERY DANGEROUS** to modify this queue directly because not only can it introduce invalid pieces, it can also change the length of the queue and lead to unexpected behavior due to constrains in the generation logic and potential piece length requirements used by playing algorithms. To clone the queue safely, use `list(game.queue)`.

### Benchmark Functions

The benchmarking functions are part of the extended API and may not be stable.

- `benchmark(alg, engine_radius: int, queue_size: int, eval_times: int = 10, print_on_completion: bool = True, print_on_move: bool = False, print_on_game: bool = False) -> tuple[float, float]`: Benchmark the given algorithm by running multiple games and printing the results.
- `compare(alg1, alg2, engine_radius: int, queue_size: int, eval_times: int = 100, print_on_completion = True, print_on_move: bool = False, print_on_game: bool = False) -> float`: Compare two algorithms by running multiple games and printing the results.
- `compare_with_data(alg, data: list[tuple[hex.HexEngine, list[hex.Piece], list[tuple[int, hex.Hex]]]], data_name: str = "data", rank_limit = None, print_on_completion = True, print_on_move: bool = False) -> float`: Compare the algorithm with a dataset of game states.

## Piece Names Mapping

The following is a mapping of piece names to their corresponding byte values:

- "uno": 8
- "full": 127
- "hallow": 119
- "triangle_3_a": 13
- "triangle_3_b": 88
- "line_3_i": 28
- "line_3_j": 73
- "line_3_k": 42
- "corner_3_i_l": 74
- "corner_3_i_r": 41
- "corner_3_j_l": 56
- "corner_3_j_r": 14
- "corner_3_k_l": 76
- "corner_3_k_r": 25
- "fan_4_a": 78
- "fan_4_b": 57
- "rhombus_4_i": 27
- "rhombus_4_j": 120
- "rhombus_4_k": 90
- "corner_4_i_l": 39
- "corner_4_i_r": 114
- "corner_4_j_l": 101
- "corner_4_j_r": 83
- "corner_4_k_l": 23
- "corner_4_k_r": 116
- "asym_4_i_a": 92
- "asym_4_i_b": 30
- "asym_4_i_c": 60
- "asym_4_i_d": 29
- "asym_4_j_a": 75
- "asym_4_j_b": 77
- "asym_4_j_c": 89
- "asym_4_j_d": 105
- "asym_4_k_a": 46
- "asym_4_k_b": 106
- "asym_4_k_c": 43
- "asym_4_k_d": 58
