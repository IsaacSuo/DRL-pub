from config.train_cfg import TrainingConfig
from .base import BasePolicy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from collections import deque
from gymnasium import Env

from model.ddqn import DoubleDeepQNetworkTagetModel, DoubleDeepQNetworkEvalModel

class DoubleDeepQNetworkPolicy(BasePolicy):
    def __init__(self, 
                 target_model: DoubleDeepQNetworkTagetModel,
                 eval_model: DoubleDeepQNetworkEvalModel):
        super().__init__()
        self.target_model = target_model
        self.eval_model = eval_model
        
    def prepare(self):
        self.init_buffer()
        
    def init_buffer(self):
        '''
        构建经验回放缓冲区‘
        '''
        self.replay_buffer = deque(maxlen=10000)
        
    def store_experience(self, state, action, reward, next_state, done):
        '''
        存储单个经验到回放缓冲区
        '''
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def sample_buffer(self, batch_size):
        '''
        在缓冲区中采样mini_batch的经验用于梯度更新
        '''
        batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        states = [self.replay_buffer[idx][0][0] for idx in batch]
        actions = [self.replay_buffer[idx][1] for idx in batch]
        rewards = [self.replay_buffer[idx][2] for idx in batch]
        next_states = [self.replay_buffer[idx][3] for idx in batch]
        dones = [self.replay_buffer[idx][4] for idx in batch]
        self.replay_buffer.clear()
        # states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[idx] for idx in batch])
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones))
    
    def step(self, 
             env: Env, 
             state, 
             action, 
             state_size, 
             total_reward, 
             train_counter, 
             ba,
             epsilon,
             gamma,
             epsilon_min,
             epsilon_decay,
             target_update_freq):
        '''
        与环境交互一步 即执行一步训练，包括前向传播、计算损失、反向传播等。
        '''
        state, action, reward, next_state, done = super().step(env, state, action, state_size, total_reward)
        # store experience into replay buffer
        # [WriteCode]
        self.store_experience(state, action, reward, next_state, done)
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
            target_q_values = []
            for s, a, r, s_, d in zip(states, actions, rewards, next_states, dones):
                if d:
                    target_q_values.append(r)
                else:
                    # Predict current Q-values for state using eval_model
                    # Use eval_model to determine best action in next_state
                    eval_y_values = self.eval_model.predict(s_, verbose=0)
                    a_best = np.argmax(eval_y_values)
                    # Use target_model to compute Q-value for that action
                    target_q_value = self.target_model.predict(s_, verbose=0)[0][a_best]
                    target_q_values.append(r + gamma * target_q_value)
            target_q_values = np.array(target_q_values)
            
            # Update only the Q-value for the taken action
            current_q_values = self.eval_model.predict(states, verbose=1)
            # print("Current Q-values before update:", current_q_values)
            for i, a in enumerate(actions):
                current_q_values[i][a] = target_q_values[i]
            current_q_values = np.array(current_q_values)
            # print("Current Q-values after update:", current_q_values)
            
            # Fit the model:
            # - Inputs: state
            # - Targets: updated Q-values (with action Q-value replaced by computed target)
            self.eval_model.model.fit(states, current_q_values, batch_size=ba, verbose=1)
            
            # Update exploration rate
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Periodically update the target network
            if train_counter % target_update_freq == 0:
                self.target_model.set_weights(self.eval_model.get_weights())