import gymnasium as gym
from gymnasium import Env
import tensorflow as tf
import numpy as np
import random

from agent.core import CoreAgent
from policy.base import BasePolicy
from config.train import TrainingConfig
from keras.callbacks import TensorBoard

# CHANGED: Import the new optimized buffer instead of using deque
from agent.replay_buffer import OptimizedReplayBuffer

class KytollyAgent(CoreAgent):
    def __init__(self, env: Env, policy: BasePolicy, cfg: TrainingConfig, cb: TensorBoard):
        super().__init__(env, policy, cfg, cb)
        self.warmup_size = cfg.warmup_size
        
        # CHANGED: Initialize the high-performance buffer
        self.replay_buffer = OptimizedReplayBuffer(
            max_size=50000,  # A reasonably large buffer size
            state_dim=self.env.observation_space.shape[0]
        )
        
    def remember(self, state, action, reward, next_state, done):
        '''Stores a single experience in the replay buffer.'''
        # CHANGED: Use the .add() method of our new buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def prepare(self):
        self.init_buffer(self.warmup_size)
    
    def init_buffer(self, min_replay_size=10000):
        '''Fills the replay buffer with initial random experiences.'''
        # CHANGED: Updated print statement for clarity
        print('Initializing optimized replay buffer, starting warmup...')
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
        
        # CHANGED: Updated print statement
        print(f"Warmup complete. Optimized replay buffer size: {len(self.replay_buffer)}")
    
    def sample_buffer(self, batch_size):
        '''Samples a mini-batch from the buffer.'''
        # CHANGED: The entire logic is now handled by our optimized buffer's .sample() method.
        # This is much cleaner and faster.
        return self.replay_buffer.sample(batch_size)
    
    def update_policy(self, experience, **kwargs):
        '''Stores experience, samples from buffer, updates policy, and decays epsilon.'''
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

        if len(self.replay_buffer) >= ba:
            train_counter += 1
            
            # Sample from the buffer and update the policy
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
        
        return state, total_reward, epsilon, train_counter
