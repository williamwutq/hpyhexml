# Top-level example code for creating a TensorFlow model for HappyHex
# This creates a convolutional neural network (CNN) model for the HappyHex game utilizing the custom HexConv and HexDynamicConv layers.
# Generally, CNNs perform better than MLPs for this application but take more time to train.
# It is designed to be run directly, not imported as a module.
# Read the console output carefully and verify the configurations before proceeding.
# This code assumes you have TensorFlow and Keras installed, and the necessary data files are placed in the correct directories.
# Adjust the paths and parameters or even add or remove layers as necessary.
# Free training and testing data is available at GitHub: `github.com/williamwutq/hpyhexml_data`
# If you have base algorithm, generate your own data with `hpyhexml.generator`.
# Copyright (c) 2025 William Wu, licensed under the MIT License.

# Parameters for training the model, replace with your own values.
save_as = 'hex_tensorflow_mlp_5_1_cnn_1.keras'
load_from = None # If you want to load a pre-trained model, specify the path here.
training_path = ['hpyhexml_data/data/train/nrsearchrank/5-1/0.txt', 
                 'hpyhexml_data/data/train/nrsearchrank/5-1/1.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/2.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/3.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/4.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/5.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/6.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/7.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/12.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/13.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/14.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/15.txt']
testing_path = ['hpyhexml_data/data/test/nrsearchrank/5-1.txt']
initial_lr = 4e-4
epoch_offset = 0 # The number of epochs already trained
epochs = 80
batch_size = 64
clipnorm = 1.2

print("\nStart training script...")
import time
import os
print(f"""
Before proceeding, ensure the following configurations are correct:\n
- TensorFlow and Keras are installed.
- hpyhex, hpyhexml, and hpyhexml/tensorflowimpl are placed in the correct directories.
- Data paths are correct (relative to this running path or absolute paths).
- Model settings and parameters are reasonable.
- Softmax uses native labels, not raw scores. Linear correlation data are within reasonable range.
- Custom loss functions are not in use or do not create sharp gradients.
- Early stopping is ENABLED unless intentionally disabled.
- {f"Load model from: {load_from}" if load_from else "Create a new model according to script"}
- Intended model save path: {save_as}
- Intended engine radius: 5
- Intended queue length: 1
- Intended top choices: <From data>
- Intended initial epoch: {epoch_offset}
- Intended epochs: {epochs}
- Intended batch size: {batch_size}
- Intended initial learning rate: {initial_lr}
""")
print("If the configurations are correct, script will execute automatically.")
print("Anytime, Press Ctrl + C to abort.")
print("Note: This script train a CNN model, which may take a long time to run. By default, auto-saving is enabled every epoch.")
print(f"      To recover from unexpected termination of the process, check keras files with name similar to {save_as}.")
print("Warning: This script contain memory leak of unknown origin, which may crush the system if run for too long.")
print("         Please monitor your system memory usage and close the process when necessary. Model is saved every epoch.")
print("Proceed to data loading [y]/n: ", end="")
response = input().strip().lower()
if response != 'y' and response:
    print("Training aborted.")
    exit(0)

if os.path.exists(save_as):
    print(f"Warning: The file '{save_as}' already exists and will be overwritten.")
    print("Do you want to proceed? y/[n]: ", end="")
    response = input().strip().lower()
    if response != 'y' or not response:
        print("Training aborted to avoid overwriting the existing file.")
        exit(0)

print("\nImporting numpy...")
import numpy as np
print("Importing hpyhex and hpyhexml...")
from hpyhex.hex import HexEngine, Piece, Hex
from hpyhexml import hex as hx
from hpyhexml.generator import load_training_data

def prepare_data(engine: HexEngine, queue: list[Piece], desired: list[tuple[int, Hex]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare the data for training the CNN model.

    Parameters:
        engine (HexEngine): The hex engine representing the game state.
        queue (list[Piece]): The queue of pieces available for placement.
        desired (list[tuple[int, Hex]]): The desired output for the model, containing tuples of piece index and position.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the input data and output data as numpy vectors.
    Raises:
        TypeError: If any of the parameters are of incorrect type.
        ValueError: If the Hex position is invalid for the given piece index for any piece in the queue.
    """
    input_data = hx.flatten_engine(engine) + hx.flatten_queue(queue)
    output_data = hx.flatten_single_desired(engine, desired, lambda x: hx.softmax_rank_score(x, len(desired)), swap_noise = 0.01, score_noise = 0)
    
    return np.array(input_data), np.array(output_data)
print("Imported modules.\n")

# Load training data
print("Loading training data...")
training_data = []
for path in training_path:
    print(f"Loading {path}...")
    training_data += load_training_data(path)
np.random.shuffle(training_data)
print(f"Loaded {len(training_data)} training samples.")
print(f"First training sample: \n{training_data[0]}\n")

# Load testing data
print("Loading testing data...")
testing_data = []
for path in testing_path:
    print(f"Loading {path}...")
    testing_data += load_training_data(path)
np.random.shuffle(testing_data)
print(f"Loaded {len(testing_data)} testing samples.")
print(f"First testing sample: \n{testing_data[0]}\n")
print()

# Parse training data
print("Parsing training data...")

x_train = []
y_train = []
for sample in training_data:
    input_vec, output_vec = prepare_data(*sample)
    x_train.append(input_vec)
    y_train.append(output_vec)

x_train = np.array(x_train)
y_train = np.array(y_train)
print(f"Parsed {len(x_train)} training samples.")
print(f"First training sample: \nInput: \n{x_train[0]}\nOutput: \n{y_train[0]}")

# Parse testing data
print("Parsing testing data...")
x_test = []
y_test = []
for sample in testing_data:
    input_vec, output_vec = prepare_data(*sample)
    x_test.append(input_vec)
    y_test.append(output_vec)

x_test = np.array(x_test)
y_test = np.array(y_test)
print(f"Parsed {len(x_test)} testing samples.")
print(f"First testing sample: \nInput: \n{x_test[0]}\nOutput: \n{y_test[0]}")
print()

# Check if data is correct
print("Data is correct and proceed to training [y]/n: ", end="")
response = input().strip().lower()
if response != 'y' and response:
    print("Training aborted.")
    exit(0)

print("\nImporting TensorFlow and Keras...")
import keras
from keras.metrics import TopKCategoricalAccuracy
from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, Reshape, Lambda, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
keras.config.enable_unsafe_deserialization()
print("Importing hpyhexml.tensorflowimpl.hexcnn...")
from hpyhexml.tensorflowimpl.hexcnn import HexConv, HexDynamicConv

# Train the model with a larger batch size for better performance
def create_model(
    grid_size=61,
    piece_size=7,
    l2_reg=1e-6,
    dropout_rate=0.04
):
        # Input layer
    input_layer = Input(shape=(grid_size + piece_size,), name="input_layer")

    # Slice grid and piece inputs
    input_grid = Lambda(lambda x: x[:, :grid_size], name="slice_grid")(input_layer)
    input_piece = Lambda(lambda x: x[:, grid_size:], name="slice_piece")(input_layer)

    # Reshape for conv input shapes
    input_grid = Reshape((grid_size, 1), name="reshape_grid")(input_grid)
    input_piece = Reshape((piece_size, 1), name="reshape_piece")(input_piece)

    # Preprocess piece with Dense layer + BatchNorm + Activation
    processed_piece = Dense(piece_size,
                            kernel_regularizer=l2(l2_reg),
                            name="dense_piece")(input_piece)
    processed_piece = BatchNormalization(name="bn_piece")(processed_piece)
    processed_piece = Activation('relu', name="act_piece")(processed_piece)

    # First HexDynamicConv block
    conv = HexDynamicConv(32, activation=None)([input_grid, processed_piece])
    conv = BatchNormalization(name="bn_conv1")(conv)
    conv = Activation('relu', name="act_conv1")(conv)
    conv = Dropout(dropout_rate / 2, name="dropout_conv1")(conv)

    # Second HexDynamicConv block
    conv = HexDynamicConv(64, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv2")(conv)
    conv = Activation('relu', name="act_conv2")(conv)
    conv = Dropout(dropout_rate, name="dropout_conv2")(conv)

    # Third HexDynamicConv block
    conv = HexDynamicConv(64, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv3")(conv)
    conv = Activation('relu', name="act_conv3")(conv)
    conv = Dropout(dropout_rate, name="dropout_conv3")(conv)

    # Fourth HexDynamicConv block
    conv = HexDynamicConv(32, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv4")(conv)
    conv = Activation('relu', name="act_conv4")(conv)
    conv = Dropout(dropout_rate / 2, name="dropout_conv4")(conv)

    # Fifth HexDynamicConv block
    conv = HexDynamicConv(12, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv5")(conv)
    conv = Activation('relu', name="act_conv5")(conv)

    # Final HexConv layer to reduce to 1 output feature per block
    conv = HexConv(1, activation=None)(conv)
    conv = BatchNormalization(name="bn_conv_final")(conv)
    conv = Activation('relu', name="act_conv_final")(conv)

    # Flatten and final output layer with softmax
    flattened = Flatten(name="flatten")(conv)
    output = Dense(grid_size, activation='softmax', name="output")(flattened)

    # Build and return model
    model = Model(inputs=input_layer, outputs=output, name="HexGridModel")

    optimizer = Adam(learning_rate=initial_lr, clipnorm=clipnorm)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy', TopKCategoricalAccuracy(k=5)])
    return model
def schedule_with_warmup_and_constant(initial_lr, epochs):
    warmup_epochs = 4 #max(1, epochs // 12)
    constant_epochs = max(1, epochs // 12)
    decay_epochs = epochs - warmup_epochs - constant_epochs

    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        elif epoch < warmup_epochs + constant_epochs:
            return initial_lr
        else:
            decay_epoch = epoch - warmup_epochs - constant_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_epoch / decay_epochs))
            return initial_lr * cosine_decay
    return lr_schedule
cosine_scheduler = keras.callbacks.LearningRateScheduler(
    schedule_with_warmup_and_constant(initial_lr, epochs)
)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
class SaveEveryEpoch(keras.callbacks.Callback):
    def __init__(self, filepath: str):
        super().__init__()
        if filepath.endswith('.keras'):
            filepath = filepath[:-6]  # Remove .keras extension for saving
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        # Save the full model
        self.model.save(f"{self.filepath}_epoch_{epoch + 1}.keras")
        print(f"\nModel saved at epoch {epoch + 1}")

if load_from:
    print(f"Loading model from {load_from} to be saved as {save_as}...")
    model = keras.models.load_model(load_from, custom_objects={"HexDynamicConv": HexDynamicConv, "HexConv": HexConv})
else:
    print(f"Creating model to be saved as {save_as}...")
    model = create_model()
print(f"Start training with {epochs} epochs and initial learning rate of {initial_lr}...")
model.fit(x_train, y_train, initial_epoch = epoch_offset, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, cosine_scheduler, SaveEveryEpoch(save_as)],)
print("Training complete.")
# Save the model for future use
model.save(save_as)
print("Start testing on test samples...")
loss = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Testing complete.")
print(f"Test Loss: {loss}\n")

# Benchmark the model
time.sleep(2)
print("Benchmarking the model...")
from hpyhex.benchmark import benchmark, compare
import algos
import time

from hpyhexml.tensorflowimpl.autoplayimpl import create_model_predictor

model_predict = create_model_predictor(save_as, "model_predict")

compare(model_predict, algos.nrsearch, 5, 1, eval_times = 1000)
benchmark(model_predict, 5, 1, eval_times = 400)
benchmark(algos.random, 5, 1, eval_times = 200)
benchmark(algos.first, 5, 1, eval_times = 200)
benchmark(algos.nrsearch, 5, 1, eval_times = 20)