import tensorflow as tf
from keras.callbacks import TensorBoard
from model.dqn_mlp import DeepQNetworkModel
from .base import BasePolicy

class DQNPolicy(BasePolicy):
    def __init__(self, model: DeepQNetworkModel, device):
        super().__init__(model, device)

    @tf.function
    def _fit_step(self,
                  states: tf.Tensor,
                  actions: tf.Tensor,
                  rewards: tf.Tensor,
                  next_states: tf.Tensor,
                  dones: tf.Tensor,
                  gamma: tf.Tensor,
                  optimizer,
                  loss_fn):
        # 1. Compute target Q-values (DQN uses max over next actions)
        next_target_q_values = self.model.target_model(next_states, training=False)  # (batch, action_dim)
        max_next_q_values = tf.reduce_max(next_target_q_values, axis=1)  # (batch,)

        q_targets = rewards + (1.0 - dones) * gamma * max_next_q_values  # (batch,)

        # 2. Compute current Q(s,a)
        with tf.GradientTape() as tape:
            q_values = self.model.online_model(states, training=True)  # (batch, action_dim)

            batch_idx = tf.range(tf.shape(actions)[0])
            action_indices = tf.stack([batch_idx, actions], axis=1)
            q_values_of_actions = tf.gather_nd(q_values, action_indices)  # (batch,)

            # 3. TD loss: MSE( Q(s,a), target )
            loss = loss_fn(q_values_of_actions, q_targets)

        # 4. Backprop
        grads = tape.gradient(loss, self.model.online_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.online_model.trainable_variables))

        return loss

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
        tensor_gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        optimizer = self.model.opt
        loss_fn = self.model.loss_fn

        loss = self._fit_step(states, actions, rewards, next_states, dones, tensor_gamma, optimizer, loss_fn)

        # periodically sync target network
        if train_counter % target_update_freq == 0:
            self.model.sync()

        return loss
