from config.train import TrainingConfig
from .base import BasePolicy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from collections import deque
from gymnasium import Env
import random
import gymnasium as gym
from gymnasium import Env
from keras.callbacks import TensorBoard
from keras.src.optimizers import Optimizer
from keras.losses import Loss
from model.ddqn_mlp import DoubleDeepQNetworkModel

class DoubleDeepQNetworkPolicy(BasePolicy):
    def __init__(self, model: DoubleDeepQNetworkModel, device):
        super().__init__(model, device)
        
    # @tf.function
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
        # Update policy using Double Deep Q-Learning update:
        # Q(s, a) = r + gamma * Q_target(S', argmax Q_eval(S', a))
        # [WriteCode]
        # states: tf.Tensor, # (32, 4)
        # actions: tf.Tensor, # (32, )
        # rewards: tf.Tensor, # (32, )
        # next_states: tf.Tensor, # (32, 4)
        # dones: tf.Tensor, # (32, )
        tensor_gamma = tf.convert_to_tensor(gamma, dtype=tf.float32) # (1, )
        optimizer = self.model.opt
        loss_fn = self.model.loss_fn
        
        loss = self._fit_step_(states, actions, rewards, next_states, dones, tensor_gamma, optimizer, loss_fn)
        # if cb:
        #     with tf.summary.create_file_writer(cb.log_dir).as_default():
        #         tf.summary.scalar('btach_loss', loss, step=train_counter)

        # Periodically update the target network
        if train_counter % target_update_freq == 0:
            self.model.sync()
            
    @tf.function
    def _fit_step_(self,
                   states: tf.Tensor,
                   actions: tf.Tensor,
                   rewards: tf.Tensor,
                   next_states: tf.Tensor,
                   dones: tf.Tensor,
                   gamma: tf.Tensor,
                   optimizer: Optimizer,
                   loss_fn: Loss,):
        # Compute target Q-values:
        # - If done, Q-target = reward (no future reward)
        # - Otherwise, Q-target = reward + gamma * Q_target(S', argmax Q_eval(S', a))
        
        # Predict current Q-values for state using eval_model
        next_online_q_values = self.model.online_model(next_states, training=False) # (32, 2)
        # Use eval_model to determine best action in next_state
        best_actions = tf.argmax(next_online_q_values, axis=1) # (32, ) eg.[0, 1, ...]
        # Use target_model to compute Q-value for that action
        next_target_q_values = self.model.target_model(next_states, training=False) # (32, 2)
        action_indices = tf.stack([
            tf.range(tf.shape(best_actions)[0]), # eg.[0, 1, ..., 31]
            tf.cast(best_actions, tf.int32) # eg.[0, 1,..., 1]  
        ], axis=1) # cast_shape (32, 2)
        # Update only the Q-value for the taken action
        # q_targets = next_target_q_values[np.arange(len(best_actions)), best_actions] * gamma + rewards
        # q_targets = np.where(dones.astype(bool), rewards, q_targets)
        q_targets_next = tf.gather_nd(next_target_q_values, action_indices) # (32, ) eg. next[0,1] = 1.1, next[1,0] = 1.4, ...
        q_targets = rewards + (1.0 - dones) * gamma * q_targets_next # (32, )
                
        # Fit the model:
        # - Inputs: state
        # - Targets: updated Q-values (with action Q-value replaced by computed target)
        
        # y_values[np.arange(actions.shape[0]), actions] = q_targets
        # self.model.online_model.fit(x=states, y=y_values, batch_size=ba, callbacks=cb, verbose=0, epochs=epoch)
        with tf.GradientTape() as tape:
            y_values = self.model.online_model(states, training=True) # (32, 2)
            actions_taken = tf.stack([
                tf.range(tf.shape(actions)[0]), # eg.[0, 1, ..., 31]
                actions
            ], axis=1) # cast_shape (32, 2)
            y_values_of_actions = tf.gather_nd(y_values, actions_taken) # (32, )
            loss = loss_fn(y_values_of_actions, q_targets)
            gradients = tape.gradient(loss, self.model.online_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.online_model.trainable_variables))

            return loss