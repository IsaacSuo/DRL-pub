import tensorflow as tf
from keras.callbacks import TensorBoard
from model.dqn_mlp import DeepQNetworkModel
from .base import BasePolicy

class DQNPolicy(BasePolicy):
    def __init__(self, model: DeepQNetworkModel, device):
        super().__init__(model, device)

    def update(self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
        gamma: float,
        train_counter: int,
        ba: int,
        target_update_freq: int,
        epoch: int,
        cb: TensorBoard,
    ):
        # Update policy using standard DQN update:
        # Q(s, a) = r + gamma * max Q_target(S', a')

        tensor_gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)

        # Compute target Q-values using DQN:
        # - If done, Q-target = reward (no future reward)
        # - Otherwise, Q-target = reward + gamma * max Q_target(S', a')

        # Use target_model to compute Q-values for next states
        next_target_q_values = self.model.target_model(next_states, training=False)  # (32, 2)
        # Take maximum Q-value across all actions
        max_next_q_values = tf.reduce_max(next_target_q_values, axis=1)  # (32,)

        # Compute TD targets
        q_targets = rewards + (1.0 - dones) * tensor_gamma * max_next_q_values  # (32,)

  
        # Create target Q-values for all actions (only update the taken action)
        current_q_values = self.model.online_model(states, training=False)  # (batch, action_dim)
        target_q_values = tf.identity(current_q_values)  # 使用tf.identity创建可写张量

        # Update only the Q-value for the taken action usingTensorFlow操作
        batch_size = tf.shape(states)[0]
        indices = tf.stack([tf.range(batch_size), actions], axis=1)  # (batch, 2)
        updates = q_targets  # (batch,)

        # 使用tf.tensor_scatter_nd_update更新Q值
        target_q_values = tf.tensor_scatter_nd_update(target_q_values, indices, updates)

        # Train using model.fit()
        history = self.model.online_model.fit(
            states,
            target_q_values,
            batch_size=ba,
            epochs=epoch,
            verbose=0,
            callbacks=[cb] if cb is not None else None
        )

        # Get the loss from history
        loss = history.history['loss'][0] if history.history and 'loss' in history.history else 0.0

        # Periodically update the target network
        if train_counter % target_update_freq == 0:
            self.model.sync()

        return loss