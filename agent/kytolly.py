import gymnasium as gym
from gymnasium import Env
from agent.core import CoreAgent
from collections import deque
import tensorflow as tf
import numpy as np
import random

from policy.base import BasePolicy
from config.train import TrainingConfig
from keras.callbacks import TensorBoard

class KytollyAgent(CoreAgent):
    def __init__(self, env: Env, policy: BasePolicy, cfg: TrainingConfig, cb: TensorBoard):
        super().__init__(env, policy, cfg, cb)
        self.replay_buffer = deque()
        
    def remember(self, state, action, reward, next_state, done):
        '''存储单个经验到回放缓冲区'''
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def prepare(self):
        self.init_buffer(1000)
    
    def init_buffer(self, min_replay_size=1000):
        '''构建经验回放缓冲区 初始存放 min_replay_size 的随机经验'''
        print('Inilize replay buffer, starting warmup...')
        state_size = self.env.observation_space.shape[0]
        state, _ = self.env.reset()
        state = np.reshape(state, [1, state_size])
        while len(self.replay_buffer) < min_replay_size:
            action = self.env.action_space.sample() 
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            done = terminated or truncated
            
            self.remember(state, action, reward, next_state, done)
            if done:
                state, _ = self.env.reset()
                state = np.reshape(state, [1, state_size])
            else:
                state = next_state
        print(f"Warmup complete. Replay buffer size: {len(self.replay_buffer)}")
    
    def sample_buffer(self, batch_size):
        '''在缓冲区中采样mini_batch的经验用于梯度更新'''
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
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        return states, actions, rewards, next_states, dones
    
    def update_policy(self, experience, **kwargs):
        '''存储经验 经验池采样 策略网络更新 探索系数衰减'''
        state, action, reward, next_state, done = experience
        total_reward = kwargs['total_reward']
        epsilon = kwargs['epsilon']
        train_counter = kwargs['train_counter']
        gamma = kwargs['gamma']
        ba = kwargs['ba']
        epsilon_min = kwargs['epsilon_min']
        epsilon_decay = kwargs['epsilon_decay']
        target_update_freq = kwargs['target_update_freq']
        epoch = kwargs['epoch']
        
        self.remember(state, action, reward, next_state, done)
        state, total_reward = next_state, total_reward + reward
        if done and len(self.replay_buffer) >= ba:
            train_counter += 1
            
            # Update policy with mini-batches if replay buffer contains enough samples
            states, actions, rewards, next_states, dones = self.sample_buffer(ba)
            self.policy.update(
                states, 
                actions, 
                rewards, 
                next_states, 
                dones, 
                gamma,
                train_counter,
                ba,
                target_update_freq,
                epoch,
                self.cb,
            )
            # Update exploration rate
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        
        # updating results follows
        return state, total_reward, epsilon, done