# Top-level example code for creating a TensorFlow model for HappyHex
# REINFORCE algorithm used to do reinforcement learning on existing models
# This script cannot and do not generate new models
# Copyright (c) 2025 William Wu, licensed under the MIT License.

# Constants
ENGINE_RADIUS = 5
QUEUE_SIZE = 3
CURRICULUM_SIZE = 1000
CURRICULUM_NAME = "random_5" #"endgame_nrsearch_5"
MODEL_PATH = "hex_tensorflow_cnn_5_3_stack_4_critic.keras"
SAVE_PATH = "hex_tensorflow_cnn_5_3_stack_4_reinforce.keras"
AGENT_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
TRAINING_EPOCHS = 100
TRAINING_PARRALLEL_GAMES = 32
TRAINING_TURN_LIMIT = 100
TRACEMALLOC  = True  # Enable memory tracing
BENCHMARKING = True  # Enable benchmarking after each epoch

# Initialization
print("Initializing...")
from hpyhex.hex import HexEngine
from algos import nrsearch
from hpyhexml.rl import curriculum
from hpyhexml.rl.game import BatchedGame
from tqdm import tqdm
empty_engine = HexEngine(ENGINE_RADIUS)

# Initialize curriculum manager with algo_based_endgame curriculum
print("Initializing curriculum manager...")
try:
    # Load an existing curriculum
    curriculum.load_curriculum(CURRICULUM_NAME, require_data=True, require_func=True)
    print(f"Curriculum {CURRICULUM_NAME} loaded.")
    # Validate the curriculum have enough engines
    existing = curriculum.count_engines_in_curriculum(CURRICULUM_NAME)
    if existing >= CURRICULUM_SIZE:
        print("Curriculum already has enough engines.")
    else:
        while existing < CURRICULUM_SIZE:
            print(f"Generating curriculum step {existing+1}")
            curriculum.generate_engine_to_curriculum(name=CURRICULUM_NAME)
            existing += 1
        curriculum.save_curriculum(CURRICULUM_NAME)
        print(f"Curriculum {CURRICULUM_NAME} saved.")
except ValueError:
    # Start a curriculum
    curriculum.random_curriculum(ENGINE_RADIUS, QUEUE_SIZE)
    #curriculum.algo_based_endgame(alg=nrsearch, radius=ENGINE_RADIUS, queue_length=QUEUE_SIZE)
    print("Curriculum created.")
    # Try load exisitng engines, if any
    try:
        from hpyhexml import generator
        engines = generator.load_engine_states(f"{CURRICULUM_NAME}.txt", print_err = True)
        if engines:
            curriculum.add_engines_to_curriculum(CURRICULUM_NAME, engines)
            print(f"Loaded {len(engines)} existing engines into curriculum.")
    except Exception | FileNotFoundError:
        pass
    for i in range(len(engines), CURRICULUM_SIZE):
        print(f"Generating curriculum step {i+1}")
        curriculum.generate_engine_to_curriculum(name=CURRICULUM_NAME)
    curriculum.save_curriculum(CURRICULUM_NAME)
    print(f"Curriculum {CURRICULUM_NAME} saved.")
print(f"Curriculum {CURRICULUM_NAME} initialized with {CURRICULUM_SIZE} engines.\n")

# Importing
print("Importing TensorFlow, Keras, and HexConv...")
import tensorflow as tf
import keras
from keras import layers
from hpyhexml.tensorflowimpl.hexcnn import HexConv, HexDynamicConv
keras.config.enable_unsafe_deserialization()
print("TensorFlow, Keras, and HexConv imported.\n")

# Inspect model layers
print("Loading model...")
model = keras.models.load_model(MODEL_PATH, custom_objects={'HexConv': HexConv, 'HexDynamicConv': HexDynamicConv})
model.summary()
# If the model does not have the right input shape, raise an error
if model.input_shape != (None, len(empty_engine) + QUEUE_SIZE * 7):
    raise ValueError(f"Model input shape {model.input_shape} does not match expected shape {(None, len(empty_engine) + QUEUE_SIZE * 7)}")
# If the model does not have the right output shape, raise an error
if model.output_shape == (None, len(empty_engine) * QUEUE_SIZE):    
    # We need a critic head, simplify by adding a dense layer with the latest convoltion layer and the output layer as input
    print("Agent-only model found, adding critic head...")
    agent = model
    x = model.layers[-2].output
    o = model.layers[-1].output
    # Flatten x and o if they are not flat
    if len(x.shape) > 2:
        x = layers.Flatten()(x)
    if len(o.shape) > 2:
        o = layers.Flatten()(o)
    # Concatenate x and o
    x = layers.Concatenate()([x, o])
    # Add a dense layer with 128 units and relu activation
    critic_intermediate = layers.Dense(128, activation='relu')(x)
    # Add the critic output layer with linear activation
    critic_output = layers.Dense(1, activation='linear', name='critic_output')(critic_intermediate)
    critic = keras.Model(inputs=model.input, outputs=critic_output)
    critic.summary()
    print("Critic head added to model.")
    # Save the modified model
    new_path = MODEL_PATH.replace(".keras", "_critic.keras")
    critic.save(new_path)
    print(f"Model saved with critic head at {new_path}.")
elif model.output_shape == (None, len(empty_engine) * QUEUE_SIZE + 1):
    print("Combined model found.")
    # Split the model into agent and critic
    agent_output = model.layers[-1].output[:, :-1]
    critic_output = model.layers[-1].output[:, -1:]
    agent = keras.Model(inputs=model.input, outputs=agent_output)
    critic = keras.Model(inputs=model.input, outputs=critic_output)
elif model.output_shape == (None, 1):
    # We only have a critic, need to build an agent from the latest convolution layer
    print("Critic-only model found, building agent from convolution layers...")
    # Get the layer called output
    output_layer = model.get_layer(name='output')
    if output_layer is None:
        raise ValueError("Model does not have an output layer named 'output' to build the agent from.")
    x = output_layer.input
    # Check if x have the correct shape
    if x.shape[1] != len(empty_engine) * QUEUE_SIZE:
        raise ValueError(f"Model output layer input shape {x.shape} does not match expected shape {(None, len(empty_engine) * QUEUE_SIZE)}")
    # Build the agent model
    agent = keras.Model(inputs=model.input, outputs=x)
    critic = model
    print("Agent model built from critic model.")
else:
    raise ValueError(f"Model output shape {model.output_shape} does not match expected shape {(None, len(empty_engine) * QUEUE_SIZE)}")
print("Model loaded.\n")

# Set up benchmarking tools
if BENCHMARKING:
    print("Setting up benchmarking tools...")
    from hpyhex.benchmark import benchmark
    from hpyhexml.tensorflowimpl.autoplayimpl import create_model_predictor
    print("Benchmarking tools set up.\n")

# Set up memory tools
if TRACEMALLOC:
    print("Setting up memory tracing...")
    import tracemalloc
    tracemalloc.start()
    print("Memory tracing set up.\n")

# Set up training
print("Setting up model trainer for training...")
from hpyhexml.tensorflowimpl.hexrl import BatchedRLTrainer
from keras.optimizers import Adam
# Build optimizers
agent_optimizer = Adam(learning_rate=AGENT_LEARNING_RATE)
critic_optimizer = Adam(learning_rate=CRITIC_LEARNING_RATE)
# Build rl trainer
trainer = BatchedRLTrainer(agent=agent, critic=critic, agent_optimizer=agent_optimizer, critic_optimizer=critic_optimizer)
print("Model configured for RL training.\n")

# Training
print("Starting training...")
# Create a batched game instance
engines = curriculum.retrieve_all_from_curriculum(CURRICULUM_NAME)
from tqdm import tqdm
for i in range(TRAINING_EPOCHS):
    print(f"Training epoch {i+1}/{TRAINING_EPOCHS}")
    policy_loss, value_loss, avg_reward = 0.0, 0.0, 0.0

    for engine in tqdm(engines, desc=f"Epoch {i+1} Engines", unit="game"):
        game = BatchedGame(TRAINING_PARRALLEL_GAMES, QUEUE_SIZE, engine)
        result = trainer.run_episode_and_update(game, TRAINING_TURN_LIMIT)
        # Collect resources
        game.interrupt()
        del game
        # Accumulate losses and rewards
        policy_loss += result['policy_loss']
        value_loss += result['value_loss']
        avg_reward += result['avg_reward']

        if TRACEMALLOC:
            # Print current memory usage and detailed distribution
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            print("[ Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)

    policy_loss /= len(engines)
    value_loss /= len(engines)
    avg_reward /= len(engines)

    print(
        f"Epoch {i+1} - Policy Loss: {policy_loss:.4f}, "
        f"Value Loss: {value_loss:.4f}, "
        f"Average Reward: {avg_reward:.4f}"
    )

    # Save the model
    temp_path = SAVE_PATH.replace(".keras", f"_epoch_{i+1}.keras")
    model.save(temp_path)
    critic_path = SAVE_PATH.replace(".keras", f"_critic_epoch_{i+1}.keras")
    critic.save(critic_path)
    print(f"Model saved at {temp_path}.\n")

    if BENCHMARKING:
        # Benchmarking
        print("Benchmarking current agent...")
        predictor = create_model_predictor(agent, "Agent")
        bench_result = benchmark(predictor, ENGINE_RADIUS, QUEUE_SIZE, 20, False, False, False)
        print(f"Benchmark results over 20 games: {bench_result}\n")

print("Training completed.")
model.save(SAVE_PATH)
print(f"Final model saved at {SAVE_PATH}.")