# Top-level example code for creating a TensorFlow model for HappyHex
# This creates a much larger model with more layers and neurons compare to run_0_tf.py with one-hot encoding data.
# It is designed to be run directly, not imported as a module.
# Read the console output carefully and verify the configurations before proceeding.
# This code assumes you have TensorFlow and Keras installed, and the necessary data files are placed in the correct directories.
# Adjust the paths and parameters as necessary.
# Free training and testing data is available at GitHub: `github.com/williamwutq/hpyhexml_data`
# If you have base algorithm, generate your own data with `hpyhexml.generator`.
# Copyright (c) 2025 William Wu, licensed under the MIT License.

# Parameters for training the model, replace with your own values.
save_as = 'hex_tensorflow_mlp_5_1_label_large_1.keras'
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
- Intended epochs: {epochs}
- Intended batch size: {batch_size}
- Intended initial learning rate: {initial_lr}
""")
print("If the configurations are correct, script will execute automatically.")
print("Anytime, Press Ctrl + C to abort.")
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
    Prepare the data for training the MLP model.

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
    output_data = hx.label_single_desired(engine, desired[0])
    
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
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

# Train the model with a larger batch size for better performance
def create_model():
    model = Sequential([
        Input(shape=(68,)),
        Dense(256, activation=None),
        BatchNormalization(),
        Activation('relu'),

        Dense(512, activation=None),
        BatchNormalization(),
        Activation('relu'),

        Dense(512, activation=None, kernel_regularizer=l2(2e-7)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.02),

        Dense(768, activation=None),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.05),

        Dense(768, activation=None, kernel_regularizer=l2(4e-7)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.05),

        Dense(768, activation=None, kernel_regularizer=l2(4e-7)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.05),

        Dense(512, activation=None),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.02),

        Dense(512, activation=None),
        BatchNormalization(),
        Activation('relu'),

        Dense(256, activation='relu'),

        Dense(61, activation='softmax')
    ])
    optimizer = Adam(learning_rate=initial_lr, clipnorm=clipnorm)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy', TopKCategoricalAccuracy(k=5)])
    return model
def schedule_with_warmup_and_constant(initial_lr, epochs):
    warmup_epochs = 2 #max(1, epochs // 12)
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
if load_from:
    print(f"Loading model from {load_from} to be saved as {save_as}...")
    model = keras.models.load_model(load_from)
else:
    print(f"Creating model to be saved as {save_as}...")
    model = create_model()
print(f"Start training with {epochs} epochs and initial learning rate of {initial_lr}...")
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, cosine_scheduler])
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