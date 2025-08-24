# Top-level example code for creating a TensorFlow model for HappyHex
# Based on run_2_tf.py, this script added multiqueue support. The model will dynamically adapt to the queue size, but the training
# and testing data must be consistent and compatible with the model.
# This creates a convolutional neural network (CNN) model for the HappyHex game utilizing the custom HexConv and HexDynamicConv layers.
# Generally, CNNs perform better than MLPs for this application but take more time to train. This is a large CNN model with many layers.
# It is designed to be run directly, not imported as a module.
# Read the console output carefully and verify the configurations before proceeding.
# This code assumes you have TensorFlow and Keras installed, and the necessary data files are placed in the correct directories.
# Adjust the paths and parameters or even add or remove layers as necessary.
# Free training and testing data is available at GitHub: `github.com/williamwutq/hpyhexml_data`
# If you have base algorithm, generate your own data with `hpyhexml.generator`.
# Copyright (c) 2025 William Wu, licensed under the MIT License.

# Parameters for training the model, replace with your own values.
save_as = 'hex_tensorflow_cnn_5_3_stack_4.keras'
load_from = None # If you want to load a pre-trained model, specify the path here.
training_path = ['hpyhexml_data/data/train/nrsearchrank/5-3/22.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/23.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/24.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/25.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/26.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/27.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/28.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/29.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/30.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/31.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/11.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/12.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/13.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/14.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/15.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/16.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/17.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/18.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/19.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/20.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-3/21.txt',
                 ]
testing_path = ['hpyhexml_data/data/test/nrsearchrank/5-3.txt',
                'hpyhexml_data/data/train/nrsearchrank/5-3/0.txt']
initial_lr = 8e-4
epoch_offset = 0 # The number of epochs already trained
epochs = 120
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
- Intended queue length: 3
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
    output_data = hx.flatten_multiple_desired(engine, queue, desired, lambda x: hx.softmax_rank_score(x, len(desired)), swap_noise = 0.00, score_noise = 0)
    #output_data = hx.label_multiple_desired(engine, queue, desired[0])
    # Changed at epoch 18
    
    return np.array(input_data), np.array(output_data)
print("Imported modules.\n")

# Load training data
print("Loading training data...")
training_data = []
for path in training_path:
    print(f"Loading {path}...")
    training_data += load_training_data(path)
np.random.shuffle(training_data)
training_data = training_data[:len(training_data) // 2]
hx.argument_queue(training_data)

print(f"Loaded {len(training_data)} training samples after cutting.")
print(f"First training sample: \n{training_data[0]}\n")

# Load testing data
print("Loading testing data...")
testing_data = []
for path in testing_path:
    print(f"Loading {path}...")
    testing_data += load_training_data(path)
np.random.shuffle(testing_data)
hx.argument_queue(testing_data)
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

import gc
del(training_data, testing_data)  # Free memory. This way we don't have large serialized lists in memory.
gc.collect()  # Collect garbage to free memory

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
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, Reshape, Permute, Lambda, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
keras.config.enable_unsafe_deserialization()
print("Importing hpyhexml.tensorflowimpl.hexcnn...")
from hpyhexml.tensorflowimpl.hexcnn import HexConv, HexDynamicConv

# Train the model with a larger batch size for better performance
def create_model(
    grid_size=61,
    piece_size=7,
    queue_size=3,
    l2_reg=1e-6,
    dropout_rate=0.04
):
    # About the design of the model:
    # - The model is designed to handle a grid of size 61 and pieces of size 7.
    # - The queue size is set to 3, meaning the model can handle 3 pieces in the queue. Change this to adapt to data.
    # - The model is split into three distinct parts:
    #   1. Input layer that slices the grid and piece inputs.
    #   2. Preprocessing of the piece inputs for each piece in the queue
    #   3. A series of HexDynamicConv layers that process the grid and piece inputs.
    #   4. A final HexConv layer to reduce the output to 3 features, representing three pieces in the queue.
    #   5. Output layer that flatten and applies softmax activation to produce the final output.
    # Input layer
    input_layer = Input(shape=(grid_size + piece_size * queue_size,), name="input_layer")

    # Slice grid and piece inputs
    input_grid = Lambda(lambda x: x[:, :grid_size], name="slice_grid")(input_layer)
    input_piece = Lambda(lambda x: x[:, grid_size:], name="slice_piece")(input_layer)

    # Reshape for conv input shapes
    input_grid = Reshape((grid_size, 1), name="reshape_grid")(input_grid)
    input_piece = Reshape((piece_size, queue_size), name="reshape_piece")(input_piece)

    # Permute piece to reuse the same weights for each piece in the queue
    # This is necessary because Dense layers expect the last dimension to be the feature dimension
    permuted_piece = Permute((2, 1), name="permute_piece")(input_piece)

    # Preprocess piece with Dense layer + BatchNorm + Activation
    processed_piece = Dense(piece_size,
                            kernel_regularizer=l2(l2_reg),
                            name="dense_piece")(permuted_piece)
    processed_piece = BatchNormalization(name="bn_piece")(processed_piece)
    processed_piece = Activation('relu', name="act_piece")(processed_piece)

    # Permute back to original shape
    processed_piece = Permute((2, 1), name="permute_back_piece")(processed_piece)

    # At this point, processed_piece has shape (batch_size, piece_size, queue_size)

    # Preprocess grid with HexConv layer
    pre_conv = HexConv(32, activation=None, name="pre_conv")(input_grid)
    pre_conv = BatchNormalization(name="bn_pre_conv")(pre_conv)
    pre_conv = Activation('relu', name="act_pre_conv")(pre_conv)

    # First HexDynamicConv block
    conv = HexDynamicConv(32, activation=None)([input_grid, processed_piece])
    conv = BatchNormalization(name="bn_conv1")(conv)
    conv = Activation('relu', name="act_conv1")(conv)

    # Second HexDynamicConv block
    conv = HexDynamicConv(64, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv2")(conv)
    conv = Activation('relu', name="act_conv2")(conv)
    conv = Dropout(dropout_rate / 4, name="dropout_conv2")(conv)

    # Third HexDynamicConv block
    conv = HexDynamicConv(64, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv3")(conv)
    conv = Activation('relu', name="act_conv3")(conv)
    conv = Dropout(dropout_rate / 2, name="dropout_conv3")(conv)

    # Fourth HexDynamicConv block
    conv = HexDynamicConv(64, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv4")(conv)
    conv = Activation('relu', name="act_conv4")(conv)
    conv = Dropout(dropout_rate, name="dropout_conv4")(conv)

    # Intermediate HexConv layer
    conv = HexConv(64, activation=None)(conv)
    conv = BatchNormalization(name="bn_conv_intermediate")(conv)
    conv = Activation('relu', name="act_conv_intermediate")(conv)
    conv = Dropout(dropout_rate, name="dropout_conv_intermediate")(conv)

    # Fifth HexDynamicConv block
    conv = HexDynamicConv(64, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv5")(conv)
    conv = Activation('relu', name="act_conv5")(conv)
    conv = Dropout(dropout_rate, name="dropout_conv5")(conv)

    # Sixth HexDynamicConv block
    conv = HexDynamicConv(64, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv6")(conv)
    conv = Activation('relu', name="act_conv6")(conv)
    conv = Dropout(dropout_rate / 2, name="dropout_conv6")(conv)

    # Seventh HexDynamicConv block
    conv = HexDynamicConv(64, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv7")(conv)
    conv = Activation('relu', name="act_conv7")(conv)
    conv = Dropout(dropout_rate / 4, name="dropout_conv7")(conv)

    # Semi-final HexConv layer
    conv = HexConv(48, activation=None)(conv)
    conv = BatchNormalization(name="bn_conv_semifinal")(conv)
    conv = Activation('relu', name="act_conv_semifinal")(conv)

    # Eighth HexDynamicConv block
    conv = HexDynamicConv(32, activation=None)([conv, processed_piece])
    conv = BatchNormalization(name="bn_conv8")(conv)
    conv = Activation('relu', name="act_conv8")(conv)

    # Final HexConv layer
    conv = HexConv(3, activation=None)(conv)
    conv = BatchNormalization(name="bn_conv_final")(conv)
    conv = Activation('relu', name="act_conv_final")(conv)

    # Flatten and final output layer with softmax
    flattened = Flatten(name="flatten")(conv)
    dense = Dense(grid_size * queue_size, activation='linear', name="dense_final")(flattened)
    output = Activation('softmax', name="output")(dense)

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
del(x_train, y_train, x_test, y_test)  # Free memory. This way we don't have large deserialized numpy arrays in memory.
gc.collect()
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