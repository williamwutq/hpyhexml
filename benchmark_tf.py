# Example code for benchmarking a TensorFlow model for HappyHex
# This is a top-level example script for benchmarking and comparing HappyHex models.
# It is designed to be run directly, not imported as a module.
# This code assumes you have a TensorFlow model and a dataset to benchmark against.
# Adjust the paths and parameters as necessary. You may choose your own algorithm and dataset for comparison.
# Free training and testing data is available at GitHub: `github.com/williamwutq/hpyhexml_data`
# If you have base algorithm, generate your own data with `hpyhexml.generator`.
# If you only want to benchmark or compare, you can comment out the other parts.
# Copyright (c) 2025 William Wu, licensed under the MIT License.

# Put your model path here
model_path = 'hpyhexml/model.keras'
# Put your data path here
data_path = 'hpyhexml_data/data/test/nrsearchrank/5-1.txt'
# Import your algorithm to compare against here
from algos import nrsearch as alg
# Number of games used for benchmarking and number of turns for comparison. Accuracy increase with more data.
benchmark_games = 10
comparison_turns = 100
# Size of engine and queue for benchmarking, must be compatible with the model.
# Incomparible sizes may result in errors or meaningless results.
engine_size = 5
queue_size = 1

if __name__ == "__main__":
    from hpyhex.benchmark import benchmark, compare, compare_with_data
    from hpyhexml.generator import load_training_data
    from hpyhexml.tensorflowimpl.autoplayimpl import create_model_predictor
    
    p = create_model_predictor(model_path, 'model_predict')

    benchmark(p, engine_size, queue_size, benchmark_games)              # Benchmark the model to see its real game performance
    compare(p, alg, engine_size, queue_size, comparison_turns)          # Compare the model to see its similarity to a known algorithm
    compare_with_data(p, load_training_data(data_path), "testing_data") # Compare the model with a dataset to see its accuracy against testing data