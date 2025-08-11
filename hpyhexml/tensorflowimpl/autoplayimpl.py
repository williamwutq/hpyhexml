import keras
from hpyhex.hex import HexEngine, Piece, Hex
import tensorflow as tf
import numpy as np
from .. import hex as hx
from .hexcnn import HexDynamicConv, HexConv

def predict_data(model: keras.Model, engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    """
    Predict the output for a given engine and queue using the trained model.

    Parameters:
        model (keras.Model): The trained Keras model for prediction.
        engine (HexEngine): The HexEngine instance representing the current state of the game.
        queue (list[Piece]): A list of Piece instances representing the queue of pieces.
    Returns:
        tuple[int, Hex]: A tuple containing the index of the piece in the queue and the predicted Hex coordinate.
    """
    input_data = hx.flatten_engine(engine) + hx.flatten_queue(queue)
    input_array = np.array(input_data)
    input_array = input_array.reshape(1, -1)  # Reshape to add batch dimension
    prediction = model(input_array, training=False).numpy()  # Call the model directly instead of using predict to avoid creation of new tensors
    # Cast it to a float array
    prediction = prediction.astype(float)[0]
    # Set impossible to -1
    possible = engine.check_positions(queue[0])
    for i in range(len(prediction)):
        if not engine.coordinate_block(i) in possible:
            prediction[i] = -1.0
    # Find the index of the maximum value in the prediction
    max_index = np.argmax(prediction)
    result_hex = engine.coordinate_block(int(max_index))
    return (0, result_hex) # we only have one piece, so the index is always 0

def create_model_predictor(model_path: str, func_name: str) -> callable:
    """
    Dynamically create a named prediction function that loads a model and calls predict_data.

    Model is loaded at creation time.

    The function created will follow the algorithm signature:
    ```python
    def algorithm(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    ```
    
    Parameters:
        model_path (str): Path to the Keras model file.
        func_name (str): Desired function name (string).
    Returns:
        function: A named function that can be used as a predictor.
    Raises:
        TypeError: If model_path or func_name is not a string.
        ValueError: If func_name is an empty string.
    """
    if not isinstance(model_path, str):
        raise TypeError("model_path must be a string")
    if not isinstance(func_name, str):
        raise TypeError("func_name must be a string")
    func_name = func_name.strip()
    if not func_name:
        raise ValueError("func_name cannot be an empty string")
    model = keras.models.load_model(model_path, custom_objects={"HexDynamicConv": HexDynamicConv, "HexConv": HexConv})
    model.call = tf.function(model.call, reduce_retracing=True) # Reduce retracing to avoid memory leaks

    def predictor(engine, queue):
        try:
            return predict_data(model, engine, queue)
        except Exception:
            raise ValueError("Prediction failed")

    # Assign function name dynamically
    predictor.__name__ = func_name

    return predictor