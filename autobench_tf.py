# Script for benchmarking a TensorFlow model for HappyHex using command line arguments
# This is a top-level example script for benchmarking and comparing HappyHex models.
# It is designed to be run directly, not imported as a module.
# This code assumes you have a TensorFlow model and a dataset to benchmark against.
# Use command line arguments to specify paths and parameters. You may choose your own algorithm and dataset for comparison.
# Free training and testing data is available at GitHub: `github.com/williamwutq/hpyhexml_data`
# If you have base algorithm, generate your own data with `hpyhexml.generator`.
# If you want a hard coded version that can be adjusted in an editor, see `benchmark_tf.py`.
# Copyright (c) 2025 William Wu, licensed under the MIT License.

__doc__ = \
'''
Benchmarking and comparing a TensorFlow model for HappyHex using command line arguments.

Available arguments:
    action: The action to perform, either 'benchmark', 'compare' (default: 'benchmark'), or * (does both actions), must be provided
    --help or -h: Show help message and exit
    --count n or -c n: Number of games used for benchmarking or number of turns for comparison (default: 10 for benchmark, 100 for compare)
    --model path, -m path, --file path, -f path, --path path, or -p path: Path to the TensorFlow model, must be provided, or benching a pre-defined algorithm
    --data path or -d path: Path to the dataset for comparison (default: None). Required if action is 'compare' and comparing with data
    --alg algorithm, --algorithm algorithm, or -a algorithm: Algorithm to compare against (default: 'nrsearch'). Options: 'nrsearch', 'random', 'nrminimax', 'first', etc.
    --engine n, -e n, --radius n, or -r n: Radius of engine for benchmarking (default: 8)
    --queue n, -q n, --size n, or -s n: Size of queue for benchmarking (default: 5)
    --printgame or -pg: Print game results during benchmarking (default: False)
    --ignoregame or -ig: Do not print game results during benchmarking (overrides --printgame) (default: True)
    --printmove or -pm: Print move results during benchmarking (default: False)
    --ignoremove or -im: Do not print move results during benchmarking (overrides --printmove) (default: True)
    --printcomplete or -pc: Print completion message after benchmarking (default: True)
    --ignorecomplete or -ic: Do not print completion message after benchmarking (overrides --printcomplete) (default: False)
'''

def error_and_exit(message: str):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)

# Command line argument parsing
import argparse, sys
parser = argparse.ArgumentParser(description='Benchmarking and comparing a TensorFlow model for HappyHex using command line arguments.')
parser.add_argument('action', type=str, choices=['benchmark', 'compare', 'all'], help="the action to perform, either 'benchmark', 'compare', or 'all' (does both actions)")
parser.add_argument('--count', '-c', type=int, default=None, help='number of games used for benchmarking or number of turns for comparison (default: 10 for benchmark, 100 for compare)')
parser.add_argument('--model', '-m', '--file', '-f', '--path', '-p', type=str, help='path to the TensorFlow model')
parser.add_argument('--data', '-d', type=str, default=None, help='path to the dataset for comparison (default: None)')
parser.add_argument('--alg', '--algorithm', '-a', type=str, default=None, help="algorithm to compare against (default: None). Options: 'nrsearch', 'random', 'nrminimax', 'first', etc.")
parser.add_argument('--engine', '-e', '--radius', '-r', type=int, help='radius of engine for benchmarking')
parser.add_argument('--queue', '-q', '--size', '-s', type=int, help='size of queue for benchmarking')
parser.add_argument('--printgame', '-pg', action='store_true', help='print game results during benchmarking (default: False)')
parser.add_argument('--ignoregame', '-ig', action='store_true', help='do not print game results during benchmarking (overrides --printgame) (default: True)')
parser.add_argument('--printmove', '-pm', action='store_true', help='print move results during benchmarking (default: False)')
parser.add_argument('--ignoremove', '-im', action='store_true', help='do not print move results during benchmarking (overrides --printmove) (default: True)')
parser.add_argument('--printcomplete', '-pc', action='store_true', help='Print completion message after benchmarking (overrides --ignorecomplete) (default: True)')
parser.add_argument('--ignorecomplete', '-ic', action='store_true', help='do not print completion message after benchmarking (default: False)')
args = parser.parse_args()

if args.action == 'benchmark':
    if args.count is None:
        args.count = 10
elif args.action == 'compare':
    if args.count is None:
        args.count = 100
elif args.action == 'all':
    if args.count is None:
        args.count = 10
else:
    error_and_exit("Invalid action. Must be 'benchmark', 'compare', or 'all'.")
if args.ignoregame:
    args.printgame = False
if args.ignoremove:
    args.printmove = False
if args.printcomplete:
    args.ignorecomplete = False
args.printcomplete = not args.ignorecomplete
if args.action in ['compare', 'all'] and args.data is None and args.alg is None:
    error_and_exit("Either data path or algorithm must be provided for comparison with --data or --alg")
if args.engine is None:
    error_and_exit("Engine radius must be provided with --engine")
if args.queue is None:
    error_and_exit("Queue size must be provided with --queue")
if args.engine < 2:
    error_and_exit("Engine radius must be at least 2")
if args.queue < 1:
    error_and_exit("Queue size must be at least 1")
if args.count < 1:
    error_and_exit("Count must be at least 1")
# Attempt to load the model
if args.model is not None:
    try:
        from hpyhexml.tensorflowimpl.autoplayimpl import create_model_predictor
        p = create_model_predictor(args.model, 'model_predict')
    except Exception as e:
        error_and_exit(f"Failed to load model from {args.model}: {e}")
if args.action in ['all', 'benchmark']:
    from hpyhex.benchmark import benchmark
    if args.model is None:
        if args.alg is None:
            error_and_exit("Either model path or algorithm must be provided for benchmarking with --model or --alg")
        from algos import __dict__ as algos_dict
        from algos import __all__ as algos_all
        if not args.alg in algos_all:
            error_and_exit(f"Algorithm '{args.alg}' not found. Available options: {', '.join(algos_all)}")
        alg_to_benchmark = algos_dict.get(args.alg.lower())
        if alg_to_benchmark is None:
            error_and_exit(f"Algorithm '{args.alg}' not found. Available options: {', '.join(algos_all)}")
        print("\nRunning benchmark...\n")
        benchmark(alg_to_benchmark, args.engine, args.queue, args.count, print_on_game=args.printgame, print_on_move=args.printmove, print_on_completion=args.printcomplete)
    else:
        print("\nRunning benchmark...\n")
        benchmark(p, args.engine, args.queue, args.count, print_on_game=args.printgame, print_on_move=args.printmove, print_on_completion=args.printcomplete)
if args.action in ['all', 'compare']:
    if args.model is None:
        error_and_exit("Model path must be provided for comparison with --model")
    from hpyhex.benchmark import compare, compare_with_data
    from hpyhexml.generator import load_training_data
    from algos import __dict__ as algos_dict
    from algos import __all__ as algos_all
    if not args.alg or not args.alg in algos_all:
        error_and_exit(f"Algorithm '{args.alg}' not found. Available options: {', '.join(algos_all)}")
    else:
        alg_to_compare = algos_dict.get(args.alg.lower())
        if alg_to_compare is None:
            error_and_exit(f"Algorithm '{args.alg}' not found. Available options: {', '.join(algos_all)}")
    if args.data is not None:
        try:
            data = load_training_data(args.data)
        except Exception as e:
            error_and_exit(f"Failed to load data from {args.data}: {e}")
        print("\nRunning comparison with data...\n")
        compare_with_data(p, data, "testing_data", print_on_completion=args.printcomplete, print_on_move=args.printmove)
    else:
        print("\nRunning comparison...\n")
        compare(p, alg_to_compare, args.engine, args.queue, args.count, print_on_completion=args.printcomplete, print_on_move=args.printmove, print_on_game=args.printgame)