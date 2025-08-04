'''
This module provides the game environment for the Hex game, including the game engine, piece factory, and game logic.
It allows for the creation of game pieces, management of the game state, and interaction with algorithms that determine moves.

Classes:

- PieceFactory: A factory class for creating and managing game pieces.
- Game: A class representing the game environment, including the game engine and piece queue.
'''


from hex import Piece, HexEngine, Hex
import random


class PieceFactory:
    '''
    PieceFactory is a utility class that provides methods for creating and managing game pieces. 
    It includes a predefined set of pieces, their corresponding byte values, and reverse mappings 
    to retrieve piece names from byte values. The class also supports generating random pieces 
    based on predefined probabilities.

    Attributes:
       pieces (dict): A dictionary mapping piece names (str) to their corresponding byte values (int).
       reverse_pieces (dict): A reverse mapping of `pieces`, mapping byte values (int) to piece names (str).
    '''

    pieces = {
        # piece_name   : byte
        "uno"          : 8,
        "full"         : 127,
        "hallow"       : 119,
        "triangle_3_a" : 13,
        "triangle_3_b" : 88,
        "line_3_i"     : 28,
        "line_3_j"     : 73,
        "line_3_k"     : 42,
        "corner_3_i_l" : 74,
        "corner_3_i_r" : 41,
        "corner_3_j_l" : 56,
        "corner_3_j_r" : 14,
        "corner_3_k_l" : 76,
        "corner_3_k_r" : 25,
        "fan_4_a"      : 78,
        "fan_4_b"      : 57,
        "rhombus_4_i"  : 27,
        "rhombus_4_j"  : 120,
        "rhombus_4_k"  : 90,
        "corner_4_i_l" : 39,
        "corner_4_i_r" : 114,
        "corner_4_j_l" : 101,
        "corner_4_j_r" : 83,
        "corner_4_k_l" : 23,
        "corner_4_k_r" : 116,
        "asym_4_i_a"   : 92,
        "asym_4_i_b"   : 30,
        "asym_4_i_c"   : 60,
        "asym_4_i_d"   : 29,
        "asym_4_j_a"   : 75,
        "asym_4_j_b"   : 77,
        "asym_4_j_c"   : 89,
        "asym_4_j_d"   : 105,
        "asym_4_k_a"   : 46,
        "asym_4_k_b"   : 106,
        "asym_4_k_c"   : 43,
        "asym_4_k_d"   : 58,
    }

    # Reverse mapping from byte value to piece name
    reverse_pieces = {v: k for k, v in pieces.items()}
    
    @staticmethod
    def get_piece(name: str) -> Piece:
        '''
        Get a piece by its name.
        Parameters:
            name (str): The name of the piece to retrieve.
        Returns:
            Piece: The piece object corresponding to the given name.
        Raises:
            ValueError: If the piece name is not found in the factory.
        '''
        if name in PieceFactory.pieces:
            return Piece(PieceFactory.pieces[name])
        else:
            raise ValueError(f"Piece '{name}' not found in factory.")
        
    @staticmethod
    def get_piece_name(p : Piece) -> str:
        '''
        Get the name of a piece based on its byte value.
        Parameters:
            p (Piece): The piece object whose name is to be retrieved.
        Returns:
            str: The name of the piece corresponding to the byte value.
        Raises:
            ValueError: If the piece byte value is not found in the factory.
        '''
        if int(p) in PieceFactory.reverse_pieces:
            return PieceFactory.reverse_pieces[int(p)]
        else:
            raise ValueError(f"Piece with byte value {int(p)} not found in factory.")
        
    @staticmethod
    def generate_piece() -> Piece:
        '''
        Generate a random piece based on frequency distribution.
        Returns:
            piece (Piece): A randomly generated piece object.
        The piece is selected based on predefined probabilities, with a mix of easier and harder pieces.
        '''
        if random.random() < 0.5:
            # Easier generation
            i = random.randint(0, 73)
            if i < 8:
                return PieceFactory.get_piece("triangle_3_a")
            elif i < 16:
                return PieceFactory.get_piece("triangle_3_b")
            elif i < 22:
                return PieceFactory.get_piece("line_3_i")
            elif i < 28:
                return PieceFactory.get_piece("line_3_j")
            elif i < 34:
                return PieceFactory.get_piece("line_3_k")
            elif i < 37:
                return PieceFactory.get_piece("corner_3_i_r")
            elif i < 40:
                return PieceFactory.get_piece("corner_3_j_r")
            elif i < 43:
                return PieceFactory.get_piece("corner_3_k_r")
            elif i < 46:
                return PieceFactory.get_piece("corner_3_i_l")
            elif i < 49:
                return PieceFactory.get_piece("corner_3_j_l")
            elif i < 52:
                return PieceFactory.get_piece("corner_3_k_l")
            elif i < 56:
                return PieceFactory.get_piece("rhombus_4_i")
            elif i < 60:
                return PieceFactory.get_piece("rhombus_4_j")
            elif i < 64:
                return PieceFactory.get_piece("rhombus_4_k")

            j = random.randint(0, 24)
            if j in {0, 1}:
                return PieceFactory.get_piece("fan_4_a")
            elif j in {2, 3}:
                return PieceFactory.get_piece("fan_4_b")
            elif j == 4:
                return PieceFactory.get_piece("corner_4_i_l")
            elif j == 5:
                return PieceFactory.get_piece("corner_4_i_r")
            elif j == 6:
                return PieceFactory.get_piece("corner_4_j_l")
            elif j == 7:
                return PieceFactory.get_piece("corner_4_j_r")
            elif j == 8:
                return PieceFactory.get_piece("corner_4_k_l")
            elif j == 9:
                return PieceFactory.get_piece("corner_4_k_r")
            elif j < 14:
                return PieceFactory.get_piece(f"asym_4_i_{chr(ord('a') + j - 10)}")
            elif j < 18:
                return PieceFactory.get_piece(f"asym_4_j_{chr(ord('a') + j - 14)}")
            elif j < 22:
                return PieceFactory.get_piece(f"asym_4_k_{chr(ord('a') + j - 18)}")
            else:
                return PieceFactory.get_piece("uno")  # Should never reach
        else:
            i = random.randint(0, 85)
            if i < 6:
                return PieceFactory.get_piece("triangle_3_a")
            elif i < 12:
                return PieceFactory.get_piece("triangle_3_b")
            elif i < 16:
                return PieceFactory.get_piece("line_3_i")
            elif i < 20:
                return PieceFactory.get_piece("line_3_j")
            elif i < 24:
                return PieceFactory.get_piece("line_3_k")
            elif i < 26:
                return PieceFactory.get_piece("corner_3_i_r")
            elif i < 28:
                return PieceFactory.get_piece("corner_3_j_r")
            elif i < 30:
                return PieceFactory.get_piece("corner_3_k_r")
            elif i < 32:
                return PieceFactory.get_piece("corner_3_i_l")
            elif i < 34:
                return PieceFactory.get_piece("corner_3_j_l")
            elif i < 36:
                return PieceFactory.get_piece("corner_3_k_l")
            elif i < 40:
                return PieceFactory.get_piece("rhombus_4_i")
            elif i < 44:
                return PieceFactory.get_piece("rhombus_4_j")
            elif i < 48:
                return PieceFactory.get_piece("rhombus_4_k")
            elif i < 54:
                return PieceFactory.get_piece("fan_4_a")
            elif i < 60:
                return PieceFactory.get_piece("fan_4_b")
            elif i < 62:
                return PieceFactory.get_piece("corner_4_i_l")
            elif i < 64:
                return PieceFactory.get_piece("corner_4_i_r")
            elif i < 66:
                return PieceFactory.get_piece("corner_4_j_l")
            elif i < 68:
                return PieceFactory.get_piece("corner_4_j_r")
            elif i < 70:
                return PieceFactory.get_piece("corner_4_k_l")
            elif i < 72:
                return PieceFactory.get_piece("corner_4_k_r")
            elif i < 76:
                return PieceFactory.get_piece(f"asym_4_i_{chr(ord('a') + i - 72)}")
            elif i < 80:
                return PieceFactory.get_piece(f"asym_4_j_{chr(ord('a') + i - 76)}")
            elif i < 84:
                return PieceFactory.get_piece(f"asym_4_k_{chr(ord('a') + i - 80)}")
            else:
                return PieceFactory.get_piece("full")
            
    def all_pieces() -> tuple[Piece]:
        '''
        Return all pieces that are defined in this factory

        Returns:
            pieces (tuple[Piece]): A list of all pieces available in the factory.
        '''
        return tuple([Piece(byte) for byte in PieceFactory.pieces.values()])


class Game:
    '''
    Game is a class that represents the game environment for Hex.
    It manages the game engine, the queue of pieces, and the game state.
    It provides methods to add pieces, make moves, and check the game status.
    Its methods are intended to catch exceptions and handle errors gracefully.

    Attributes:
        engine (HexEngine): The game engine that manages the game state.
        queue (list[Piece]): The queue of pieces available for placement.
        result (tuple[int, int]): The current result of the game, including the score and turn number.
        score (int): The current score of the game.
        turn (int): The current turn number in the game.
        end (bool): Whether the game has ended.
    '''
    def __init__(self, r: int, q: int) -> None:
        '''
        Initialize the game with a game engine of radius r and game queue of length q.

        Parameters:
            r (int): The radius of the hexagonal game board.
            q (int): The length of the queue for pieces.
        Raises:
            ValueError: If r is less than 2 or q is less than 1.
        '''
        if r < 2:
            raise ValueError("Radius must be greater than or equals two")
        if q < 1:
            raise ValueError("Queue length must be at least one")
        self.__engine = HexEngine(r)
        self.__queue = [PieceFactory.generate_piece() for _ in range(q)]
        self.__score = 0
        self.__turn = 0
        self.__end = False
    
    def add_piece(self, piece_index: int, coord: Hex) -> bool:
        '''
        Add a piece to the game engine at the specified coordinates.
        Parameters:
            piece_index (int): The index of the piece in the queue to be added.
            coord (Hex): The coordinates where the piece should be placed.
        Returns:
            bool: True if the piece was successfully added, False otherwise.
        '''
        # Check piece exists
        if not isinstance(piece_index, int) or not isinstance(coord, Hex):
            return False
        if piece_index < 0 or piece_index >= len(self.__queue):
            return False
        piece = self.__queue[piece_index]
        # Add piece to engine and increment score and turn
        try:
            self.__engine.add_piece(coord, piece)
            self.__score += len(piece)
        except ValueError:
            return False
        self.__queue[piece_index] = PieceFactory.generate_piece()
        self.__score += len(self.__engine.eliminate())
        self.__turn += 1
        # Check whether the game has ended
        for p in PieceFactory.all_pieces():
            if self.__engine.check_positions(p):
                break
        else:
            self.__end = True
        # Return true if added
        return True
    
    def make_move(self, algorithm) -> bool:
        '''
        Make a move using the specified algorithm.
        Parameters:
            algorithm (callable): The algorithm to use for making the move.
                The algorithm should follow the signature: `algorithm(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]`.
        Returns:
            bool: True if the move was successfully made, False otherwise.
        '''
        if self.__end:
            return False
        try:
            index, coord = algorithm(self.__engine, self.__queue)
        except Exception:
            return False
        if not isinstance(index, int) or not isinstance(coord, Hex):
            return False
        return self.add_piece(index, coord)
    
    def __str__(self):
        '''
        Return a string representation of the game state.
        Returns:
            str: A string representation of the game state,
                including engine, queue, score, turn, and whether the game has ended.
        '''
        return f"Game(engine={self.__engine}, queue={self.__queue}, score={self.__score}, turn={self.__turn}, end={self.__end})"
    
    def __repr__(self) -> str:
        '''
        Return a string representation of the game state.
        Returns:
            str: A string representation of the game state.
        '''
        return f"({self.__engine}, {self.__queue})"
    
    @property
    def end(self) -> bool:
        '''
        Returns whether this game has ended.
        Returns:
            is_end (bool): True if the game has ended, False otherwise.
        '''
        return self.__end

    @property
    def result(self) -> tuple[int, int]:
        '''
        Returns the current result of this game.
        Returns:
            result (tuple[int, int]): A tuple containing the current turn number and score, in the order (turn, score).
        '''
        return self.__turn, self.__score
    
    @property
    def turn(self) -> int:
        '''
        Returns the current turn number of this game.
        Returns:
            turn (int): The current turn number in the game.
        '''
        return self.__turn
    
    @property
    def score(self) -> int:
        '''
        Returns the current score of this game.
        Returns:
            score (int): The current score in the game.
        '''
        return self.__score
    
    @property
    def engine(self) -> HexEngine:
        '''
        Returns the referene to game engine of this game.
        Returns:
            engine (HexEngine): The HexEngine instance used in this game.
        '''
        return self.__engine
    
    @property
    def queue(self) -> list[Piece]:
        '''
        Returns the referene to the queue of pieces available in this game.
        Returns:
            queue (list[Piece]): The list of pieces currently in the queue.
        '''
        return self.__queue