'''
A package containing various algorithms for the Hex game.

These algorithms can be used for training, benchmarking, and playing the game.
The algorithms use the following signature:
```python
def algorithm(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    ...
```

All algorithms may raise a `ValueError` if no valid moves are found, or inputs are invalid.
'''

from .base import random, first
from .nr import nrsearch, nrminimax, nrminimaxmix, nrminimaxeven, nrentropy, nrsearchworst, nrscoreindex, nrnaivescoreindex, nrdenseindex, nrbineliminate
from .rc import rcminimax

__all__ = ['random', 'first', 'nrminimax', 'nrminimaxmix', 'nrminimaxeven', 'nrentropy', 'nrsearch',
           'nrdenseindex', 'nrscoreindex', 'nrnaivescoreindex', 'nrbineliminate', 'nrsearchworst', 'rcminimax']