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

import time, sys # Should not fail
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
    input_data_b = Piece.vec_to_numpy_float32_stacked(queue)
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
from hpyhexml.pytorchimpl.hexcnn import HexDense, HexShrink, PureHexConv, MaskedHexConv

# Model Definition
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.num_cells_5 = HexEngine.solve_length(5)
        self.num_cells_3 = HexEngine.solve_length(3)

        self.conv1 = MaskedHexConv(in_channels=1, out_channels=64, kernel_in=1, kernel_radius=2, engine_radius=5)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.pureconv1 = PureHexConv(in_channels=64, out_channels=128, kernel_radius=2, engine_radius=5)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.conv2 = MaskedHexConv(in_channels=128, out_channels=256, kernel_in=1, kernel_radius=2, engine_radius=5)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.shrink = HexShrink(engine_radius=5, shrink_by=2) # To radius 3
        self.conv4 = MaskedHexConv(in_channels=256, out_channels=512, kernel_in=1, kernel_radius=2, engine_radius=3)
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.dense1 = HexDense(in_channels=512, out_channels=1024, engine_radius=3)
        self.output = nn.Linear(1024 * self.num_cells_3, self.num_cells_5)


    def forward(self, x_a, x_b):
        x = self.conv1(x_a, x_b)
        x = x.view(-1, 64, self.num_cells_5)
        x = F.relu(self.batchnorm1(x))
        x = x.view(-1, 64 * self.num_cells_5)
        
        x = self.pureconv1(x)
        x = x.view(-1, 128, self.num_cells_5)
        x = F.relu(self.batchnorm2(x))
        x = x.view(-1, 128 * self.num_cells_5)
        
        x = self.conv2(x, x_b)
        x = x.view(-1, 256, self.num_cells_5)
        x = F.relu(self.batchnorm3(x))
        x = x.view(-1, 256 * self.num_cells_5)
        
        x = x.view(-1, 256, self.num_cells_5)
        x = self.shrink(x)
        x = x.view(-1, 256 * self.num_cells_3)
        
        x = self.conv4(x, x_b)
        x = x.view(-1, 512, self.num_cells_3)
        x = F.relu(self.batchnorm6(x))
        x = x.view(-1, 512 * self.num_cells_3)
        
        x = self.dense1(x)
        x = x.view(x.size(0), -1)  # Flatten for dense layer
        x = self.output(x)
        return x
    

# Cosine Decay Learning Rate Scheduler
def cosine_decay_scheduler(optimizer, initial_lr, epochs):
    def lr_lambda(epoch):
        return 0.5 * (1 + torch.cos(torch.tensor(epoch / epochs * 3.141592653589793)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Load or Create Model
if load_from:
    print(f"Loading model from {load_from} to be saved as {save_as}...")
    model = torch.load(load_from)
else:
    print(f"Creating model to be saved as {save_as}...")
    model = Model()

# Get the device
# Check for CUDA (NVIDIA GPUs)
if torch.cuda.is_available():
    device = torch.device("cuda")
# Check for MPS (Apple Silicon GPUs)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
# Check for DirectML (Intel Arc, Intel iGPUs, AMD GPUs)
else:
    try:
        import torch_directml
        dml_device = torch_directml.device()
        device = dml_device
    except ImportError:
        # Fallback to CPU if no GPU is available
        device = torch.device("cpu")

print(f"Using device: {device}")
model.to(device)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = cosine_decay_scheduler(optimizer, initial_lr, epochs)
criterion = nn.CrossEntropyLoss()

# Prepare DataLoader
train_dataset = TensorDataset(x_train_a, x_train_b, y_train)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(x_test_a, x_test_b, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training Loop
print(f"Start training with {epochs} epochs and initial learning rate of {initial_lr}...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    for batch_idx, (a, b, targets) in enumerate(train_loader, 1):
        a, b, targets = a.to(device), b.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(a, b)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Elementary TUI: progress bar and running loss
        progress = int(40 * batch_idx / total_batches)
        bar = '[' + '=' * progress + ' ' * (40 - progress) + ']'
        avg_loss = running_loss / batch_idx
        sys.stdout.write(f'\r{bar} {batch_idx}/{total_batches} - Running Loss: {avg_loss:.4f}')
        sys.stdout.flush()
    print()  # Move to next line after epoch
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for inputs_a, inputs_b, targets in val_loader:
            inputs_a, inputs_b, targets = inputs_a.to(device), inputs_b.to(device), targets.to(device)
            outputs = model(inputs_a, inputs_b)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, pred_top1 = outputs.topk(1, dim=1)
            _, pred_top5 = outputs.topk(5, dim=1)
            correct_top1 += (pred_top1.squeeze() == targets).sum().item()
            correct_top5 += (pred_top5 == targets.unsqueeze(1)).sum().item()
            total += targets.size(0)
    
    val_loss /= len(val_loader)
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {running_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Top1 Acc: {top1_acc:.4f} - Top5 Acc: {top5_acc:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= 4:
            print("Early stopping triggered.")
            break
    scheduler.step()

model.load_state_dict(best_model_state)
print("Training complete.")
# Save model
torch.save(model, save_as)

