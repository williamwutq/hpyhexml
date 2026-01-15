import tensorflow as tf
from .. import hex as hx
import numpy as np

'''
Contains a simple batched RL training loop using BatchedGame, an agent (policy) and a critic (value network).
Uses REINFORCE with a baseline from the critic. Built for multi-piece queues.
'''

class BatchedRLTrainer:
    '''
    A simple batched RL training loop using BatchedGame, an agent (policy) and a critic (value network).
    Uses REINFORCE with a baseline from the critic. Built for multi-piece queues.
    '''

    def __init__(self, agent, critic, agent_optimizer, critic_optimizer, gamma=0.99):
        '''
        Initializes the trainer with a batched game environment, agent and critic functions, optimizers, and discount factor.

        Parameters:
            agent (tf.keras.Model): Model taking (tf.Tensor) -> action logits (tf.Tensor of shape [batch, num_actions]).
            critic (tf.keras.Model): Model taking (tf.Tensor) -> value predictions (tf.Tensor of shape [batch]).
            agent_optimizer (tf.keras.optimizers.Optimizer)
            critic_optimizer (tf.keras.optimizers.Optimizer)
            gamma (float): Discount factor.
        '''
        self.agent = agent
        self.critic = critic
        self.agent_optimizer = agent_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma

    def run_episode_and_update(self, env, limit=None):
        '''
        Runs a batch of games, collects trajectories, computes losses, and applies gradients.

        Parameters:
            env (BatchedGame): The batched game environment to run.
            limit (int, optional): Optional limit on the number of turns per game. If None, runs until all games finish.
        Returns:
            dict: Dictionary with keys "policy_loss", "value_loss", and "avg_reward"
        '''
        if not callable(env):
            raise ValueError("env must be a callable BatchedGame instance.")
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        masks = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        obs_inputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        taken_actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, infer_shape=False)

        idx = 0

        def algorithm_wrapper(engines, queues):
            nonlocal idx
            nonlocal taken_actions, obs_inputs, values
            # Policy forward pass
            inputs = self.__env_to_input(engines, queues)
            logits = self.agent(inputs, training=True)
            action_dist = tf.random.categorical(logits, 1, dtype=tf.int32)
            action_dist = tf.squeeze(action_dist, axis=1)
            # Cache inputs and actions
            obs_inputs = obs_inputs.write(idx, inputs)
            taken_actions = taken_actions.write(idx, action_dist)
            # Value prediction
            value_pred = self.critic(inputs, training=True)
            # This is shaped [batch, 1], convert to [batch]
            value_pred = tf.squeeze(value_pred, axis=1)
            values = values.write(idx, value_pred)
            # Return moves
            moves = self.__action_to_moves(action_dist, engines, queues)
            # Increment step index
            idx += 1
            return moves

        def feedback_wrapper(results):
            nonlocal idx
            nonlocal rewards, masks
            # results: list of (delta_turn, delta_score, total_moves)
            batch_rewards = tf.constant([r[1] for r in results], dtype=tf.float32)
            rewards = rewards.write(idx - 1, batch_rewards)
            done_mask = tf.constant([0.0 if r[0] == 0 else self.gamma **idx for r in results], dtype=tf.float32)
            masks = masks.write(idx - 1, done_mask)

        # Run batched game once
        env(algorithm_wrapper, feedback_wrapper, limit=limit)

        # Convert lists to tensors
        obs_inputs = obs_inputs.concat()
        taken_actions = taken_actions.concat()
        values = values.concat()
        rewards = rewards.concat()
        masks = masks.concat()
        # After env run and computing returns/advantages:
        returns = rewards * masks
        returns = tf.stop_gradient(returns)

        advantages = returns - values  # keep values from rollout for now
    
        # Get shapes
        T, _ = obs_inputs.shape

        del values

        # Update policy
        with tf.GradientTape() as tape:
            # Forward pass with flattened inputs
            logits = self.__minibatch_forward(self.agent, obs_inputs, batch_size=256)  # [T*B, num_actions]
            logp_all = tf.nn.log_softmax(logits, axis=-1)
            # Gather log-probs of taken actions
            chosen_logp = tf.gather(logp_all, taken_actions, batch_dims=1)  # [T]
            policy_loss = -tf.reduce_mean(chosen_logp * advantages)

        del logits, logp_all, chosen_logp

        grads = tape.gradient(policy_loss, self.agent.trainable_variables)
        self.agent_optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))

        # Critic update (MSE loss on returns)
        with tf.GradientTape() as tape:
            # Forward pass with flattened inputs
            value_preds = self.__minibatch_forward(self.critic, obs_inputs, batch_size=256)  # [T*B, 1]
            value_loss = tf.reduce_mean(tf.square(value_preds - returns))
        
        del value_preds

        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        
        del obs_inputs, taken_actions, returns, advantages

        # Compute losses and average reward
        return_policy_loss = float(policy_loss)
        return_value_loss = float(value_loss)
        avg_reward = float(tf.reduce_mean(rewards))

        del tape, grads, policy_loss, value_loss, rewards
        del env
        tf.keras.backend.clear_session()

        return {
            "policy_loss": return_policy_loss,
            "value_loss": return_value_loss,
            "avg_reward": avg_reward
        }

    def __minibatch_forward(self, model, inputs, batch_size):
        '''
        Run a model forward pass in smaller minibatches to save memory.
        '''
        outputs = []
        for i in range(0, inputs.shape[0], batch_size):
            batch = inputs[i:i + batch_size]
            outputs.append(model(batch, training=True))
        return tf.concat(outputs, axis=0)

    @tf.function
    def __compute_returns(self, rewards, masks, gamma):
        '''
        Compute discounted returns with episode masking.
            Works with lists of tensors of varying shapes.
        Each rewards[t] and masks[t] must have the same shape.
        '''
        T = rewards.size()
        returns = tf.TensorArray(
            dtype=tf.float32,
            size=T,
            infer_shape=False  # allow ragged elements
        )

        # Iterate in reverse
        for t in tf.range(T - 1, -1, -1):
            r_t = rewards.read(int(t))
            m_t = masks.read(int(t))
        
            returns = returns.write(t, gamma * m_t + r_t)

        return returns.concat()
    
    def __action_to_moves(self, action_indices, engines, queues):
        '''
        Convert discrete action indices to (piece_index, Hex position) for a batch of games.
        No masking of impossible moves.
        
        Parameters:
            action_indices (tf.Tensor or np.ndarray): shape [batch], each entry is a flat action index.
            queues (list[list[Piece]]): batch of piece queues for each game.
            engines (list[HexEngine]): batch of engines for each game.
    
        Returns:
            list[tuple[int, Hex]]: list of moves in the (piece_index, Hex) format expected by BatchedGame.
        '''
        moves = []
        for batch_idx, action in enumerate(action_indices.numpy()):
            result_hex = engines[batch_idx].coordinate_block(int(action) % len(engines[batch_idx]))
            piece_index = int(action) // len(engines[batch_idx])
            moves.append((piece_index, result_hex))
        return moves
    
    def __env_to_input(self, engines, queues):
        '''
        Convert the environment state to input for the agent and critic.
        
        Parameters:
            engines (list[HexEngine]): List of HexEngine instances.
            queues (list[list[Piece]]): List of piece queues.
        
        Returns:
            tuple: (engine_inputs, queue_inputs)
        '''
        data_list = []
        for i in range(len(engines)):
            flat_data = hx.flatten_engine(engines[i]) + hx.flatten_queue(queues[i])
            data_list.append(flat_data)
        batch_np = np.stack(data_list, axis=0).astype(np.float32)
        return tf.convert_to_tensor(batch_np)