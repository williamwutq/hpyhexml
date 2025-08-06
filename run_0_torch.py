# Top-level example code for creating a PyTorch model for HappyHex
# It is designed to be run directly, not imported as a module.
# Read the console output carefully and verify the configurations before proceeding.
# This code assumes you have torch installed, and the necessary data files are placed in the correct directories.
# Adjust the paths and parameters as necessary.
# Free training and testing data is available at GitHub: `github.com/williamwutq/hpyhexml_data`
# If you have base algorithm, generate your own data with `hpyhexml.generator`.
# Copyright (c) 2025 William Wu, licensed under the MIT License.

# Parameters for training the model, replace with your own values.
save_as = 'hex_tensorflow_mlp_5_1_label_1.pt'
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
import torch
print("Parsing training data...")

x_train = []
y_train = []
for sample in training_data:
    input_vec, output_vec = prepare_data(*sample)
    x_train.append(input_vec)
    y_train.append(output_vec)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_train = torch.argmax(y_train, dim=1)
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
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
y_test = torch.argmax(y_test, dim=1)
print(f"Parsed {len(x_test)} testing samples.")
print(f"First testing sample: \nInput: \n{x_test[0]}\nOutput: \n{y_test[0]}")
print()

# Check if data is correct
print("Data is correct and proceed to training [y]/n: ", end="")
response = input().strip().lower()
if response != 'y' and response:
    print("Training aborted.")
    exit(0)

print("\nImporting PyTorch...")
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Model Definition
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batchnorm0 = nn.BatchNorm1d(68)
        self.dense1 = nn.Linear(68, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        
        self.dense2 = nn.Linear(128, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        
        self.dense3 = nn.Linear(256, 384)
        self.batchnorm3 = nn.BatchNorm1d(384)
        
        self.dense4 = nn.Linear(384, 384)
        self.batchnorm4 = nn.BatchNorm1d(384)
        
        self.dense5 = nn.Linear(384, 256)
        self.batchnorm5 = nn.BatchNorm1d(256)
        
        self.dense6 = nn.Linear(256, 122)
        self.dense7 = nn.Linear(122, 61)
        
        self.dropout = nn.Dropout(0.05)
        self.l2_reg = 2e-7

    def forward(self, x):
        x = self.batchnorm0(x)
        x = F.relu(self.batchnorm1(self.dense1(x)))
        x = F.relu(self.batchnorm2(self.dense2(x)))
        
        x = self.dense3(x)
        x = x + self.l2_reg * torch.sum(x ** 2)  # L2 Regularization approximation
        x = F.relu(self.batchnorm3(x))
        x = self.dropout(x)
        
        x = self.dense4(x)
        x = x + self.l2_reg * torch.sum(x ** 2)
        x = F.relu(self.batchnorm4(x))
        x = self.dropout(x)
        
        x = F.relu(self.batchnorm5(self.dense5(x)))
        x = F.relu(self.dense6(x))
        x = F.softmax(self.dense7(x), dim=1)
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
model.to(device)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = cosine_decay_scheduler(optimizer, initial_lr, epochs)
criterion = nn.CrossEntropyLoss()

# Prepare DataLoaders
dataset = TensorDataset(x_train, y_train)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training Loop
print(f"Start training with {epochs} epochs and initial learning rate of {initial_lr}...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
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

# Testing
print("Start testing on test samples...")
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
test_loss /= len(test_loader)
print("Testing complete.")
print(f"Test Loss: {test_loss}\n")
