from config.train_cfg import TrainingConfig
from .base import BasePolicy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from collections import deque
from gymnasium import Env
import random

from model.ddqn import DoubleDeepQNetworkTagetModel, DoubleDeepQNetworkOnlineModel

class DoubleDeepQNetworkPolicy(BasePolicy):
    def __init__(self, 
                 target_model: DoubleDeepQNetworkTagetModel,
                 online_model: DoubleDeepQNetworkOnlineModel):
        super().__init__()
        self.target_model = target_model
        self.online_model = online_model
    
    def predict(self, state, verbose):
        '''
        重写基类 predict 方法， 指定用 online_model 预测
        在 self.act 中使用
        '''
        self.online_model.predict(state, verbose)
        
    def prepare(self):
        self.init_buffer()
        
    def init_buffer(self):
        '''
        构建经验回放缓冲区‘
        '''
        self.replay_buffer = deque()
        
    def store_experience(self, state, action, reward, next_state, done):
        '''
        存储单个经验到回放缓冲区
        '''
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def sample_buffer(self, batch_size):
        '''
        在缓冲区中采样mini_batch的经验用于梯度更新
        '''
        # Ensure we have enough samples
        assert len(self.replay_buffer) >= batch_size, (
            f"Not enough samples in buffer to sample {batch_size} items.")

        # Sample a mini-batch
        minibatch = random.sample(self.replay_buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states, dtype=np.float32).squeeze()
        next_states = np.array(next_states, dtype=np.float32).squeeze()
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones
        
    def step(self,
             env: Env,
             state_size,
             state,
             action,
             total_reward,
             train_counter,
             cb,
             epsilon,
             ba,
             gamma,
             epsilon_min,
             epsilon_decay,
             target_update_freq,
             epoch,
             ):
        '''
        与环境交互一步 即执行一步训练，包括前向传播、计算损失、反向传播等。
        '''
        next_state, reward, done = super().step(env, action, state_size)
        self.store_experience(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            return state, total_reward, epsilon, done
        
        if len(self.replay_buffer) >= ba:
            train_counter += 1
            # Update policy with mini-batches if replay buffer contains enough samples
            # Update policy using Double Deep Q-Learning update:
            # Q(s, a) = r + gamma * Q_target(S', argmax Q_eval(S', a))
            # [WriteCode]
            states, actions, rewards, next_states, dones = self.sample_buffer(ba)
            
            # Compute target Q-values:
            # - If done, Q-target = reward (no future reward)
            # - Otherwise, Q-target = reward + gamma * Q_target(S', argmax Q_eval(S', a))
            
            # Predict current Q-values for state using eval_model
            online_q_values = self.online_model.predict(next_states, verbose=0)
            # Use eval_model to determine best action in next_state
            best_actions = np.argmax(online_q_values, axis=1)
             # Use target_model to compute Q-value for that action
            target_q_values = self.target_model.predict(next_states, verbose=0) 
            q_targets = target_q_values[np.arange(len(best_actions)), best_actions] * gamma + rewards
            # Update only the Q-value for the taken action
            q_targets = np.where(dones.astype(bool), rewards, q_targets)
            
            # Fit the model:
            # - Inputs: state
            # - Targets: updated Q-values (with action Q-value replaced by computed target)
            y_values = self.online_model.predict(states, verbose=0)
            y_values[np.arange(actions.shape[0]), actions] = q_targets
            self.online_model.fit(X=states, Y=y_values, batch_size=ba, callbacks=cb, verbose=0, epoch=epoch)
            

            # Update exploration rate
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Periodically update the target network
            if train_counter % target_update_freq == 0:
                self.target_model.set_weights(self.online_model.get_weights())
                
        return state, total_reward, epsilon, done
    
    def evaluate(self, max_timesteps=500):
        return super().evaluate(self.online_model, max_timesteps)