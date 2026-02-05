# Top-level example code for creating a PyTorch model for HappyHex
# It is designed to be run directly, not imported as a module.
# Read the console output carefully and verify the configurations before proceeding.
# This code assumes you have torch installed, and the necessary data files are placed in the correct directories.
# Adjust the paths and parameters as necessary.
# The current script requires the hyphex-rs package to be installed and accessible, and the hpyhex package to be uninstalled.
# You can install hyphex-rs via pip: `pip install hyphex-rs` for python version 3.8 to 3.12.
# Free training and testing data is available at GitHub: `github.com/williamwutq/hpyhexml_data`
# If you have base algorithm, generate your own data with `hpyhexml.generator`.
# Copyright (c) 2026 William Wu, licensed under the MIT License.

# Parameters for training the model, replace with your own values.
save_as = 'hex_torch_hexcnn_5_1_label_2.pt'
load_from = None # If you want to load a pre-trained model, specify the path here.
training_path = ['hpyhexml_data/data/train/nrsearchrank/5-1/0.txt', 
                 'hpyhexml_data/data/train/nrsearchrank/5-1/1.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/2.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/3.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/4.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/5.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/6.txt',
                 'hpyhexml_data/data/train/nrsearchrank/5-1/7.txt']
testing_path = ['hpyhexml_data/data/test/nrsearchrank/5-1.txt']
initial_lr = 1e-3
epochs = 100
batch_size = 64
clipnorm = 0.8

print("\nStart training script...")
print(f"""
Before proceeding, ensure the following configurations are correct:\n
- Torch and hyphex-rs are installed and accessible.
- hpyhexml, and hpyhexml/torchimpl are placed in the correct directories.
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

import time # Should not fail
print("\nImporting numpy...")
try:
    import numpy as np
except ImportError:
    print("Error: numpy is not installed.")
    print("Please install numpy via `pip install numpy`.")
    exit(1)
print("Importing hpyhex (hpyhex-rs)...")
try:
    from hpyhex import HexEngine, Piece, Hex
except ImportError:
    print("Error: hpyhex-rs is not installed or not accessible.")
    print("This script requires hpyhex-rs acceleration, which is not provided by hpyhex.")
    print("Please install hpyhex-rs via `pip install hyphex-rs` and ensure hpyhex is uninstalled.")
    exit(1)
print("Importing torch...")
try:
    import torch
except ImportError:
    print("Error: torch is not installed.")
    print("Please install torch via `pip install torch`.")
    exit(1)
print("Importing hpyhexml...")
from hpyhexml import hex_rs as hx
from hpyhexml.generator import load_training_data
print("Imported modules.\n")

# Load training data
print("Loading training data...")
current_time = time.perf_counter()
training_data = []
for path in training_path:
    print(f"Loading {path}...")
    training_data += load_training_data(path)
np.random.shuffle(training_data)
print(f"Loaded {len(training_data)} training samples in {time.perf_counter() - current_time:.2f} seconds.")
print(f"First training sample: \n{training_data[0]}\n")

# Load testing data
print("Loading testing data...")
current_time = time.perf_counter()
testing_data = []
for path in testing_path:
    print(f"Loading {path}...")
    testing_data += load_training_data(path)
np.random.shuffle(testing_data)
print(f"Loaded {len(testing_data)} testing samples in {time.perf_counter() - current_time:.2f} seconds.")
print(f"First testing sample: \n{testing_data[0]}\n")
print()

# Parse training data
print("Parsing training data...")
current_time = time.perf_counter()
def prepare_data(engine: HexEngine, queue: list[Piece], desired: list[tuple[int, Hex]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_data_a = hx.flatten_engine(engine),
    input_data_b = Piece.vec_to_numpy_float32_flat(queue) # Honestly, here should be stacked, but since queue is length 1, it's the same.
    output_data = hx.flatten_single_desired(engine, desired, lambda x: hx.softmax_rank_score(x, len(desired)))
    # Performance benchmarking shows that flatten_single_desired_optimized is actually slower for sparse desired lists.

    return input_data_a, input_data_b, output_data


x_train_a = []
x_train_b = []
y_train = []
for sample in training_data:
    a, b, output_vec = prepare_data(*sample)
    x_train_a.append(a)
    x_train_b.append(b)
    y_train.append(output_vec)

x_train_a = torch.tensor(np.array(x_train_a), dtype=torch.float32)
x_train_b = torch.tensor(np.array(x_train_b), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
y_train = torch.argmax(y_train, dim=1)
print(f"Parsed {len(x_train_a)} training samples in {time.perf_counter() - current_time:.2f} seconds.")
print(f"First training sample: \nInput: \n{x_train_a[0], x_train_b[0]}\nOutput: \n{y_train[0]}")

# Parse testing data
print("Parsing testing data...")
current_time = time.perf_counter()
x_test_a = []
x_test_b = []
y_test = []
for sample in testing_data:
    a, b, output_vec = prepare_data(*sample)
    x_test_a.append(a)
    x_test_b.append(b)
    y_test.append(output_vec)

x_test_a = torch.tensor(np.array(x_test_a), dtype=torch.float32)
x_test_b = torch.tensor(np.array(x_test_b), dtype=torch.float32)
y_test = torch.tensor(np.array(y_test), dtype=torch.float32)
y_test = torch.argmax(y_test, dim=1)
print(f"Parsed {len(x_test_a)} testing samples in {time.perf_counter() - current_time:.2f} seconds.")
print(f"First testing sample: \nInput: \n{x_test_a[0], x_test_b[0]}\nOutput: \n{y_test[0]}")
print()


print("\nImporting PyTorch...")
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split