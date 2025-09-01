# Example code for generating training data for HappyHex machine learning models.
# This is a top-level example script for generating training data, usable for any models.
# It is designed to be run directly, not imported as a module.
# Adjust the paths and parameters as necessary. You may choose or write your own algorithm for generation.
# Copyright (c) 2025 William Wu, licensed under the MIT License.

'''
A simple wrapper script to hpyhexml.generator module to generate training data for HappyHex machine learning models.

To use this script, enter the desired parameters and run it. The script will generate training data files
in the specified directory. The generated data can be used for training machine learning models.

To obtain the generated data, you can use the load_training_data function from hpyhexml.generator module.

To write your own ranking algorithm, you can import the hpyhex module and follow the following signature:

```python
def algorithm(engine: HexEngine, queue: list[Piece], significant_choices: int) -> list[tuple[int, Hex]]
```

The algorithm should return a list of tuples, where each tuple contains an integer index to the piece and a Hex position.
The list should be in the order of the ranking, with the most significant choice first. The algorithm should return
at most `significant_choices` choices.
'''

# Settings
total_files = 4                  # Total number of files to generate
samples_per_file = 100000        # Number of samples per file
engine_radius = 5                # Radius of the hexagonal engine
queue_size = 3                   # Size of the engine and queue for the game
significant_choices = 9          # Number of significant choices to generate per sample
file_directory = 'mydata'        # Directory to save the generated files
file_name = 'trainingdata'       # Base name for the files
naming_offset = 0                # Offset number for file naming, useful for multiple runs

# Advanced settings
alg = None                       # Your own algorithm to use for generating training data, None for default
remove_head  = 0.05              # Percentage of samples to remove from the head of the game turns
remove_tail  = 0.05              # Percentage of samples to remove from the tail of the game turns
move_dropout = 0.1               # Percentage of moves to randomly drop from total game turns
verbose      = True              # Whether to print verbose output during generation

if __name__ == "__main__":
    print("\nRunning generation script...\n")
    # Type checks
    if not isinstance(total_files, int):
        print("Error: total_files must be an integer.")
        exit(1)
    if not isinstance(samples_per_file, int):
        print("Error: samples_per_file must be an integer.")
        exit(1)
    if total_files <= 0:
        print("No file to generate.")
        exit(0)
    if samples_per_file <= 0:
        print("No sample to generate.")
        exit(0)
    if not isinstance(engine_radius, int) or engine_radius < 2:
        print("Error: engine_radius must be an integer greater than or equal to 2.")
        exit(1)
    if not isinstance(queue_size, int) or queue_size < 1:
        print("Error: queue_size must be an integer greater than or equal to 1.")
        exit(1)
    if not isinstance(significant_choices, int) or significant_choices < 1:
        print("Error: significant_choices must be an integer greater than or equal to 1.")
        exit(1)
    if not isinstance(file_directory, str):
        file_directory = str(file_directory)
    if not isinstance(file_name, str):
        file_name = str(file_name)
    if not isinstance(naming_offset, int):
        print("Error: naming_offset must be an integer.")
    if naming_offset < 0:
        print("Naming offset is negative, are you sure to use it? [y]/n: ", end="")
        response = input().strip().lower()
        if response != 'y' and response:
            print("Abort generation.")
            exit(0)
    if not isinstance(remove_head, float) or not isinstance(remove_tail, float):
        print("Error: remove_head and remove_tail must be floats.")
        exit(1)
    if remove_head < 0 or remove_tail < 0 or remove_head > 1 or remove_tail > 1 or remove_head + remove_tail >= 1:
        print("Error: remove_head and remove_tail must be between 0 and 1 and their sum should not exceed 1.")
        exit(1)
    if not isinstance(move_dropout, float):
        print("Error: move_dropout must be a float.")
        exit(1)
    if move_dropout < 0 or move_dropout > 1:
        print("Error: move_dropout must be between 0 and 1.")
        exit(1)
    if not isinstance(verbose, bool):
        verbose = bool(verbose)

    # Try to find directory
    import os
    if not os.path.exists(file_directory):
        print(f"Directory '{file_directory}' does not exist. Do you want to create it? [y]/n: ", end="")
        response = input().strip().lower()
        if response == 'y' or not response:
            os.makedirs(file_directory)
            print(f"    Directory '{file_directory}' created.")
        else:
            print("    Directory not found. Change directory before restart the script.")
            exit(0)

    # Verify settings
    print(f"\nVerifying settings:\n")
    print(f"    Total files to generate: {total_files}")
    print(f"    Samples per file: {samples_per_file}")
    print(f"    Engine radius: {engine_radius}")
    print(f"    Queue size: {queue_size}")
    print(f"    Significant choices: {significant_choices}")
    print(f"    File directory: {file_directory}")
    print(f"    File name: {file_name}")
    print(f"    Naming offset: {naming_offset}")
    print(f"    Algorithm: {'Default' if alg is None else alg.__name__}")
    print(f"    Remove head percentage: {remove_head}")
    print(f"    Remove tail percentage: {remove_tail}")
    print(f"    Move dropout percentage: {move_dropout}")
    print(f"    Verbose output: {verbose}")
    print("\nAre the settings correct? [y]/n: ", end="")
    response = input().strip().lower()
    if response != 'y' and response:
        print("Generation aborted.")
        exit(0)

    # Start generation
    print(f"\nStart generation of {total_files} files with {samples_per_file} samples each.")
    print("To abort, press Ctrl + C at any time.")
    from hpyhexml.generator import generate_training_data_limited, save_training_dataset
    for i in range(total_files):
        print(f"Generating file {i + 1}/{total_files}...")
        try:
            if alg is not None:
                data = generate_training_data_limited(
                    samples_per_file,
                    algorithm=alg,
                    engine_radius=engine_radius,
                    queue_size=queue_size,
                    significant_choices=significant_choices,
                    remove_head=remove_head,
                    remove_tail=remove_tail,
                    move_dropout=move_dropout,
                    verbose=verbose
                )
            else:
                data = generate_training_data_limited(
                    samples_per_file,
                    engine_radius=engine_radius,
                    queue_size=queue_size,
                    significant_choices=significant_choices,
                    remove_head=remove_head,
                    remove_tail=remove_tail,
                    move_dropout=move_dropout,
                    verbose=verbose
                )
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
        file_path = f"{file_directory}/{file_name}_{engine_radius}_{queue_size}_{str(naming_offset + i)}.txt"
        if os.path.exists(file_path):
            print(f"File '{file_path}' already exists. Do you want to overwrite it? [y]/n: ", end="")
            response = input().strip().lower()
            if response != 'y' and response:
                print(f"Skipping file {file_path}.")
                continue
        save_training_dataset(data, file_path, print_err=True)
        print(f"File {i + 1}/{total_files} saved to {file_path}.")
    print("\nGeneration completed.")