import torch
import numpy as np
from hpyhex.hex import HexEngine, Piece, Hex
from .. import hex as hx

# Check for CUDA (NVIDIA GPUs)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
# Check for MPS (Apple Silicon GPUs)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
# Check for DirectML (Intel Arc, Intel iGPUs, AMD GPUs)
else:
    try:
        import torch_directml
        dml_device = torch_directml.device()
        DEVICE = dml_device
    except ImportError:
        # Fallback to CPU if no GPU is available
        DEVICE = torch.device("cpu")

def predict_data(model: torch.nn.Module, engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    """
    Predict the output for a given engine and queue using the trained PyTorch model.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model for prediction.
        engine (HexEngine): The HexEngine instance representing the current state of the game.
        queue (list[Piece]): A list of Piece instances representing the queue of pieces.
    Returns:
        tuple[int, Hex]: A tuple containing the index of the piece in the queue and the predicted Hex coordinate.
    """
    input_data = hx.flatten_engine(engine) + hx.flatten_queue(queue)
    input_array = np.array(input_data, dtype=np.float32)
    input_tensor = torch.tensor(input_array).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).squeeze(0).cpu().numpy()  # Remove batch dimension

    # Set impossible positions to -1
    possible = engine.check_positions(queue[0])
    for i in range(len(prediction)):
        if not engine.coordinate_block(i) in possible:
            prediction[i] = -1.0

    max_index = np.argmax(prediction)
    result_hex = engine.coordinate_block(int(max_index))
    return (0, result_hex)  # we only have one piece, so index is always 0

def create_model_predictor(model_path: str, func_name: str) -> callable:
    """
    Dynamically create a named prediction function that loads a PyTorch model and calls predict_data.

    Model is loaded at creation time.

    The function created will follow the algorithm signature:
    ```python
    def algorithm(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]:
    ```
    
    Parameters:
        model_path (str): Path to the PyTorch model file (.pt or .pth).
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

    # Load the PyTorch model
    model = torch.load(model_path, map_location=DEVICE)
    model.to(DEVICE)

    def predictor(engine, queue):
        try:
            return predict_data(model, engine, queue)
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    predictor.__name__ = func_name
    return predictor