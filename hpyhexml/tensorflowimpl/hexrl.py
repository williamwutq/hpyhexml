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
        log_probs = []
        values = []
        rewards = []
        masks = []

        def algorithm_wrapper(engines, queues):
            # Policy forward pass
            inputs = self.__env_to_input(engines, queues)
            logits = self.agent(inputs, training=True)
            action_dist = tf.random.categorical(logits, 1)
            action_dist = tf.squeeze(action_dist, axis=1)
            # Log-prob for policy loss
            logp_all = tf.nn.log_softmax(logits)
            chosen_logp = tf.gather(logp_all, action_dist, batch_dims=1)
            log_probs.append(chosen_logp)
            # Value prediction
            value_pred = self.critic(inputs, training=True)
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

        # Convert lists to tensors
        log_probs = tf.stack(log_probs)        # [T, batch]
        values = tf.stack(values)              # [T, batch]
        rewards = tf.stack(rewards)            # [T, batch]
        masks = tf.stack(masks)                # [T, batch]

        # Compute returns
        returns = self.__compute_returns(rewards, masks, self.gamma)
        returns = tf.stop_gradient(returns)

        # Advantage
        advantages = returns - values

        # Update policy
        with tf.GradientTape() as tape:
            policy_loss = -tf.reduce_mean(log_probs * advantages)
        grads = tape.gradient(policy_loss, self.agent.trainable_variables)
        self.agent_optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))

        # Update critic (MSE loss on returns)
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean(tf.square(values - returns))
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
        returns = tf.transpose(returns.stack(), perm=[1, 0])
        return returns
    
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
            result_hex = engines[batch_idx].coordinate_block(int(action))
            moves.append((0, result_hex)) # single-piece queue, so always 0
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