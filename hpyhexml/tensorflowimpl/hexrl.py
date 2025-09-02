import tensorflow as tf
from .. import hex as hx

class SinglePieceBatchedRLTrainer:
    '''
    A simple batched RL training loop using BatchedGame, an agent (policy) and a critic (value network).
    Uses REINFORCE with a baseline from the critic. Built for single-piece queues or single-piece virtual queues.
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
        values = []
        rewards = []
        masks = []
        obs_inputs = []
        taken_actions = []

        def algorithm_wrapper(engines, queues):
            # Policy forward pass
            inputs = self.__env_to_input(engines, queues)
            logits = self.agent(inputs, training=True)
            action_dist = tf.random.categorical(logits, 1)
            action_dist = tf.squeeze(action_dist, axis=1)
            # Cache inputs and actions
            obs_inputs.append(inputs)
            taken_actions.append(action_dist)
            # Value prediction
            value_pred = self.critic(inputs, training=True)
            # This is shaped [batch, 1], convert to [batch]
            value_pred = tf.squeeze(value_pred, axis=1)
            values.append(value_pred)
            # Return moves
            moves = self.__action_to_moves(action_dist, engines, queues)
            return moves

        def feedback_wrapper(results):
            # results: list of (delta_turn, delta_score, total_moves)
            batch_rewards = tf.constant([r[1] for r in results], dtype=tf.float32)
            rewards.append(batch_rewards)
            done_mask = tf.constant([0.0 if r[0] == 0 else 1.0 for r in results], dtype=tf.float32)
            masks.append(done_mask)

        # Run batched game once
        env(algorithm_wrapper, feedback_wrapper, limit=limit)

        # Expand size to [T, batch], fill empty with 0
        def pad_to_max_T(tensor_list):
            padded = []
            for t in tensor_list:
                pad_size = env.max_batch_size - t.shape[0]
                if pad_size > 0:
                    padding = tf.zeros([pad_size] + t.shape[1:].as_list(), dtype=t.dtype)
                    padded.append(tf.concat([t, padding], axis=0))
                else:
                    padded.append(t)
            return padded
        obs_inputs = pad_to_max_T(obs_inputs)
        taken_actions = pad_to_max_T(taken_actions)
        values = pad_to_max_T(values)
        rewards = pad_to_max_T(rewards)
        masks = pad_to_max_T(masks)

        # Convert lists to tensors
        obs_inputs = tf.stack(obs_inputs)      # [T, batch, input_dim]
        taken_actions = tf.stack(taken_actions)  # [T, batch]
        values = tf.stack(values)              # [T, batch]
        rewards = tf.stack(rewards)
        masks = tf.stack(masks)

        # After env run and computing returns/advantages:
        returns = self.__compute_returns(rewards, masks, self.gamma)
        returns = tf.stop_gradient(returns)
        advantages = returns - values  # keep values from rollout for now
    
        # Flatten [T, B, input_dim] → [T*B, input_dim]
        T, B, input_dim = obs_inputs.shape
        flat_inputs = tf.reshape(obs_inputs, [T * B, input_dim])
        flat_actions = tf.reshape(taken_actions, [T * B])

        # Update policy
        with tf.GradientTape() as tape:
            # Forward pass with flattened inputs
            logits = self.agent(flat_inputs, training=True)  # [T*B, num_actions]
            logp_all = tf.nn.log_softmax(logits, axis=-1)    
            # Gather log-probs of taken actions
            chosen_logp = tf.gather(logp_all, flat_actions, batch_dims=1)  # [T*B]
            chosen_logp = tf.reshape(chosen_logp, [T, B])  # back to [T, B]
            policy_loss = -tf.reduce_mean(chosen_logp * advantages)

        grads = tape.gradient(policy_loss, self.agent.trainable_variables)
        self.agent_optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))

        # Critic update (MSE loss on returns)
        with tf.GradientTape() as tape:
            # Forward pass with flattened inputs
            value_preds = self.critic(flat_inputs, training=True)   # [T*B, 1] or [T*B]
            value_preds = tf.squeeze(value_preds, axis=-1)          # [T*B]
            value_preds = tf.reshape(value_preds, [T, B])           # [T, B]
            value_loss = tf.reduce_mean(tf.square(value_preds - returns))
            grads = tape.gradient(value_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "avg_reward": float(tf.reduce_mean(rewards))
        }

    def __compute_returns(self, rewards, masks, gamma):
        '''
        Compute discounted returns with episode masking.
        '''
        T, B = rewards.shape
        returns = tf.TensorArray(dtype=tf.float32, size=T)
        running_return = tf.zeros([B], dtype=tf.float32)
        for t in reversed(range(T)):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns = returns.write(t, running_return)
        return returns.stack()
    
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
        tensor_inputs = []
        for i in range(len(engines)):
            data = hx.flatten_engine(engines[i]) + hx.flatten_queue(queues[i])
            # Convert to tensor
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            tensor_inputs.append(data)
        # Stack all inputs into a single tensor
        return tf.stack(tensor_inputs, axis=0)  # Shape: [batch, input_size]