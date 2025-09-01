# Simple script for generating training data for HappyHex machine learning models using command line inputs.
# This is a top-level example script for generating training data, usable for any models.
# It is designed to be run directly, not imported as a module.
# Use command line arguments to specify paths and parameters. Pathes can either be relative or absolute.
# You may choose or write your own algorithm for generation, provide that they are importable.
# Copyright (c) 2025 William Wu, licensed under the MIT License.

__doc__ = \
'''
A simple wrapper script to hpyhexml.generator module to generate training data for HappyHex machine learning models.

To use this script, run from command line. The script will generate training data files
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

import argparse, sys

def error_and_exit(message: str):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='A simple wrapper script to generate training data for HappyHex machine learning models.',
                                     epilog = "Detailed documentation: \n" + __doc__ + "\nExample usage: \n python generate.py "
                                     "--total_files 2 --samples_per_file 50000 --engine 5 --queue 3 --significant_choices 9 --directory mydata --file trainingdata --verbose")
    parser.add_argument('--total_files', '-tf', '-fs', type=int, default=1, help='total number of files to generate (default: 1)')
    parser.add_argument('--samples_per_file', '--samples', '-spf', '-c',type=int, default=100000, help='number of samples per file (default: 100000)')
    parser.add_argument('--alg', '--algorithm', '-a', type=str, default=None, help="algorithm for generation (default: None, which will use nrsearch)")
    parser.add_argument('--engine', '-e', '--radius', '-r', type=int, help='radius of engine for generation')
    parser.add_argument('--queue', '-q', '--size', '-s', type=int, help='size of queue for generation')
    parser.add_argument('--significant_choices', '-sc', type=int, help='number of significant choices to generate per sample (default: 1)')
    parser.add_argument('--directory', '--dir', '-d', type=str, default=None, help='directory to save the generated files (default: mydata)')
    parser.add_argument('--file', '-f', '--name', '-n', type=str,default=None, help='base name for the files (default: trainingdata)')
    parser.add_argument('--naming_offset', '-no', type=int, default=0, help='offset number for file naming, useful for multiple runs (default: 0)')
    parser.add_argument('--regex', '-rx', type=str, default="{dir}/{name}_{engine}_{queue}_{offset}.txt", help='regex for file naming, use {dir} for directory, {name} for base name, {engine} for engine radius, {queue} for queue size, and {offset} for offset respectively (default: {dir}/{name}_{engine}_{queue}_{offset}.txt)')
    parser.add_argument('--remove_head', '-rh', type=float, default=0.05, help='percentage of samples to remove from the head of the game turns (default: 0.05)')
    parser.add_argument('--remove_tail', '-rt', type=float, default=0.05, help='percentage of samples to remove from the tail of the game turns (default: 0.05)')
    parser.add_argument('--move_dropout', '-md', type=float, default=0.1, help='percentage of moves to randomly drop from total game turns (default: 0.1)')
    parser.add_argument('--verbose', '-v', action='store_true', help='print verbose output during generation (default: False)')
    args = parser.parse_args()
    
    total_files = args.total_files
    if total_files < 0:
        error_and_exit("total_files must be a non-negative integer.")
    samples_per_file = args.samples_per_file
    if samples_per_file < 0:
        error_and_exit("samples_per_file must be a non-negative integer.")
    if args.engine is not None:
        engine_radius = args.engine
    else:
        error_and_exit("Engine radius must be specified.")
    if engine_radius < 2:
        error_and_exit("engine_radius must be greater than or equal to 2.")
    if args.queue is not None:
        queue_size = args.queue
    else:
        error_and_exit("Queue size must be specified.")
    if queue_size < 1:
        error_and_exit("queue_size must be greater than or equal to 1.")
    if args.significant_choices is not None:
        significant_choices = args.significant_choices
    else:
        significant_choices = 1
    if significant_choices < 1:
        error_and_exit("significant_choices must be greater than or equal to 1.")
    if args.directory is not None:
        file_directory = args.directory
    else:
        error_and_exit("Directory must be specified.")
    if args.file is not None:
        file_name = args.file
    else:
        error_and_exit("File name must be specified.")
    naming_offset = args.naming_offset
    if naming_offset < 0:
        print("Naming offset is negative, are you sure to use it? [y]/n: ", end="")
        response = input().strip().lower()
        if response != 'y' and response:
            print("Abort generation.")
            exit(0)
    alg_name = args.alg
    regex = args.regex
    remove_head = args.remove_head
    remove_tail = args.remove_tail
    move_dropout = args.move_dropout
    if remove_head < 0 or remove_tail < 0 or remove_head > 1 or remove_tail > 1 or remove_head + remove_tail >= 1:
        error_and_exit("remove_head and remove_tail must be between 0 and 1 and their sum should not exceed 1.")
    if move_dropout < 0 or move_dropout > 1:
        error_and_exit("move_dropout must be between 0 and 1.")
    verbose = args.verbose
    # Check for regex validity (contain only the required fields, no extra fields but maybe less fields)
    allowed_fields = ['dir', 'name', 'engine', 'queue', 'offset']
    import re
    fields_in_regex = re.findall(r'\{(.*?)\}', regex)
    for field in fields_in_regex:
        if field not in allowed_fields:
            error_and_exit(f"Invalid field '{field}' in regex. Allowed fields are: {', '.join(allowed_fields)}")
    for field in allowed_fields:
        if field not in fields_in_regex:
            print(f"Warning: Field '{field}' not found in regex. The corresponding value will not be included in the file name.")
    import string
    try:
        # This will raise a ValueError if there are mismatched or invalid braces
        for _, field_name, _, _ in string.Formatter().parse(regex):
            pass
    except ValueError as e:
        error_and_exit(f"Invalid format string in regex: {e}")
    
    # Import algorithm if provided
    if alg_name is not None:
        # Importantly, these algorithms are not provided by algos module, so we cannot use it. Now we ask for an import
        import importlib
        # Split by dot to get module and function
        if '.' in alg_name:
            module_name, func_name = alg_name.rsplit('.', 1)
        else:
            module_name, func_name = 'algos', alg_name
        try:
            module = importlib.import_module(module_name)
            alg = getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            error_and_exit(f"Failed to import algorithm '{alg_name}': {e}")
        # Verify the algorithm signature
        import inspect
        sig = inspect.signature(alg)
        params = sig.parameters
        if len(params) != 3:
            error_and_exit(f"Algorithm '{alg_name}' must have exactly 3 parameters: engine, queue, significant_choices.")
        param_names = list(params.keys())
        if param_names[0] != 'engine' or param_names[1] != 'queue' or param_names[2] != 'significant_choices':
            error_and_exit(f"Algorithm '{alg_name}' must have parameters named: engine, queue, significant_choices.")
    else:
        alg = None

    print("\nRunning generation script...\n")

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

    # Start generation
    print(f"\nStart generation of {total_files} files with {samples_per_file} samples each.")
    print("To abort, press Ctrl + C at any time.")
    from hpyhexml.generator import generate_training_data, save_training_dataset
    for i in range(total_files):
        print(f"Generating file {i + 1}/{total_files}...")
        try:
            if alg is not None:
                data = generate_training_data(
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
                data = generate_training_data(
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
            error_and_exit(f"{e}")
        file_path = regex.format(dir = file_directory, name = file_name, engine = engine_radius, queue = queue_size, offset = naming_offset + i)
        if os.path.exists(file_path):
            print(f"File '{file_path}' already exists. Do you want to overwrite it? [y]/n: ", end="")
            response = input().strip().lower()
            if response != 'y' and response:
                print(f"Skipping file {file_path}.")
                continue
        save_training_dataset(data, file_path, print_err=True)
        print(f"File {i + 1}/{total_files} saved to {file_path}.")
    print("\nGeneration completed.")