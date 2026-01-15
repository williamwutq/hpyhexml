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

## Quickstart

1. Install the package:
   ```bash
   pip install hpyhex
   ```
2. Import and use the main classes as shown above.
3. Create custom algorithms to interact with the game environment.

## Benchmarking Algorithms

The package includes benchmarking tools to evaluate and compare algorithms:

```python
from hpyhex.benchmark import benchmark, compare

# Benchmark a single algorithm
avg_score, avg_turn = benchmark(simple_algorithm, engine_radius=3, queue_size=5, eval_times=10)

# Compare two algorithms
similarity = compare(alg1, alg2, engine_radius=3, queue_size=5, eval_times=100)
```
